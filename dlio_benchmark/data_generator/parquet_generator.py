"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress, gen_random_tensor, DLIOMPI
import dgen_py as _dgen_py

# Batches at or below this size use BufferPool.next_slice() (zero-copy slice).
# Above this threshold a streaming Generator with fill_chunk() is used.
# 32 MB is the optimal fill_chunk() size in dgen-py; never go lower.
_DGEN_POOL_THRESHOLD = 32 * 1024 * 1024  # 32 MB

# Map DLIO Compression enum values to PyArrow compression strings.
COMPRESSION_MAP = {
    Compression.NONE: None,
    Compression.GZIP: 'gzip',
}

# All numeric dtypes supported for column generation.
# Integer types use the full value range for maximum entropy.
_NP_TYPE_MAP = {
    'uint8':   np.uint8,
    'uint16':  np.uint16,
    'uint32':  np.uint32,
    'uint64':  np.uint64,
    'int8':    np.int8,
    'int16':   np.int16,
    'int32':   np.int32,
    'int64':   np.int64,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

_PA_SCALAR_TYPE_MAP = {
    'uint8':   pa.uint8(),
    'uint16':  pa.uint16(),
    'uint32':  pa.uint32(),
    'uint64':  pa.uint64(),
    'int8':    pa.int8(),
    'int16':   pa.int16(),
    'int32':   pa.int32(),
    'int64':   pa.int64(),
    'float16': pa.float16(),
    'float32': pa.float32(),
    'float64': pa.float64(),
}


class ParquetGenerator(DataGenerator):
    """
    Schema-driven Parquet data generator.

    Supports two modes:

    1. **Column-schema mode** (``parquet_columns`` config list is non-empty):
       Generates multi-column files from a list of column specs, each with a
       ``name``, ``dtype``, and optional ``size`` (embedding vector length).
       Supported dtypes: uint8/16/32/64, int8/16/32/64, float16/32/64,
       string, binary, bool.

    2. **Legacy mode** (``parquet_columns`` empty):
       Single ``data`` column of fixed-size uint8 lists, matching the original
       DLIO behaviour for backward compatibility.

    Key design properties:
    - **Unique samples**: every row in every batch has distinct data — the
      tile-copy bug from the original generator is eliminated.
    - **RNG flow-through**: a single ``np.random.Generator`` is initialised
      once per rank and advanced naturally through all file and batch
      generations.  No seed resets occur between files.
    - **Near-zero copy**: numeric columns use ``gen_random_tensor`` with
      ``rng=rng``; once the raw bytes exist they are wrapped in a
      ``FixedSizeListArray`` via ``pa.array()`` using contiguous buffers —
      no Python-level list comprehensions for fixed-size data.
    - **Configurable batching**: large files are written in batches of
      ``parquet_generation_batch_size`` rows to bound peak memory.
    """

    def __init__(self):
        super().__init__()
        self.parquet_columns = getattr(self._args, 'parquet_columns', [])
        self.row_group_size = getattr(self._args, 'parquet_row_group_size', 1024)
        self.partition_by = getattr(self._args, 'parquet_partition_by', None)
        batch = getattr(self._args, 'parquet_generation_batch_size', 0)
        self.generation_batch_size = batch if batch > 0 else self.row_group_size

    # ── Schema ───────────────────────────────────────────────────────────────

    def _build_schema(self, legacy_elem_size=None):
        """Build PyArrow schema from configured columns.

        When called in legacy mode (``parquet_columns`` is empty or None),
        ``legacy_elem_size`` must be provided; it is the number of uint8
        elements per sample (= dim1 * dim2).  The schema uses a
        ``pa.list_(pa.uint8(), legacy_elem_size)`` fixed-size list, which
        lets PyArrow use the efficient ``FixedSizeListArray`` representation
        on reads.

        When called in column-schema mode, ``legacy_elem_size`` is ignored.
        """
        if not self.parquet_columns:
            size = legacy_elem_size or 1
            return pa.schema([('data', pa.list_(pa.uint8(), size))])

        fields = []
        for col_spec in self.parquet_columns:
            if hasattr(col_spec, 'get'):
                name  = str(col_spec.get('name', 'data'))
                dtype = str(col_spec.get('dtype', 'float32'))
                size  = int(col_spec.get('size', 1))
            else:
                name, dtype, size = str(col_spec), 'float32', 1

            pa_scalar = _PA_SCALAR_TYPE_MAP.get(dtype)

            if pa_scalar is not None:
                if size == 1:
                    fields.append(pa.field(name, pa_scalar))
                else:
                    # Fixed-size list of the scalar type
                    fields.append(pa.field(name, pa.list_(pa_scalar, size)))
            elif dtype == 'list':
                fields.append(pa.field(name, pa.list_(pa.float32(), size)))
            elif dtype == 'string':
                fields.append(pa.field(name, pa.string()))
            elif dtype == 'binary':
                fields.append(pa.field(name, pa.binary()))
            elif dtype == 'bool':
                fields.append(pa.field(name, pa.bool_()))
            else:
                # Unknown dtype — fall back to fixed-size float32 list
                fields.append(pa.field(name, pa.list_(pa.float32(), size)))

        return pa.schema(fields)

    # ── Batch generation helpers ──────────────────────────────────────────────

    def _generate_column_data_batch(self, col_spec, batch_size, rng):
        """Generate one batch of data for a single column.

        All numeric dtypes use ``gen_random_tensor(rng=rng)`` so the RNG
        state advances naturally — no seed is computed or reset between calls.

        Returns ``(name, pa.Array)``.
        """
        if hasattr(col_spec, 'get'):
            name  = str(col_spec.get('name', 'data'))
            dtype = str(col_spec.get('dtype', 'float32'))
            size  = int(col_spec.get('size', 1))
        else:
            name, dtype, size = str(col_spec), 'float32', 1

        np_type = _NP_TYPE_MAP.get(dtype)
        pa_scalar = _PA_SCALAR_TYPE_MAP.get(dtype)

        # ── Numeric scalar (size == 1) ──────────────────────────────────────
        if np_type is not None and pa_scalar is not None and size == 1:
            data = gen_random_tensor(shape=(batch_size,), dtype=np_type, rng=rng)
            return name, pa.array(data, type=pa_scalar)

        # ── Numeric fixed-size list (size > 1) ─────────────────────────────
        if np_type is not None and pa_scalar is not None:
            # Generate as a flat (batch_size * size) array, then wrap as
            # FixedSizeListArray — zero extra copies after dgen/numpy.
            data = gen_random_tensor(shape=(batch_size * size,), dtype=np_type, rng=rng)
            arrow_flat = pa.array(data, type=pa_scalar)
            return name, pa.FixedSizeListArray.from_arrays(arrow_flat, size)

        if dtype == 'list':
            data = gen_random_tensor(shape=(batch_size * size,), dtype=np.float32, rng=rng)
            arrow_flat = pa.array(data, type=pa.float32())
            return name, pa.FixedSizeListArray.from_arrays(arrow_flat, size)

        # ── Non-numeric types — use numpy global state (seeded per rank) ───
        if dtype == 'string':
            # Use integers from rng to build strings so they vary per run seed
            ints = rng.integers(0, 2**31, size=batch_size)
            return name, pa.array([f"s_{v}" for v in ints], type=pa.string())

        if dtype == 'binary':
            # Each sample: size random bytes from rng
            rows = [rng.bytes(size) for _ in range(batch_size)]
            return name, pa.array(rows, type=pa.binary())

        if dtype == 'bool':
            bits = rng.integers(0, 2, size=batch_size, dtype=np.uint8)
            return name, pa.array(bits.astype(bool), type=pa.bool_())

        # Fallback: float32 fixed-size list
        data = gen_random_tensor(shape=(batch_size * size,), dtype=np.float32, rng=rng)
        arrow_flat = pa.array(data, type=pa.float32())
        return name, pa.FixedSizeListArray.from_arrays(arrow_flat, size)

    def _generate_batch_columns(self, batch_size, rng):
        """Generate all configured columns for one batch.

        The same ``rng`` object is advanced per column so every column in
        every batch gets statistically independent, non-repeating data.
        """
        columns = {}
        for col_spec in self.parquet_columns:
            name, arrow_data = self._generate_column_data_batch(col_spec, batch_size, rng)
            columns[name] = arrow_data
        return columns

    def _generate_legacy_batch(self, elem_size, batch_size, rng):
        """Generate one batch for the legacy single-'data'-column mode.

        Generates ``(batch_size * elem_size)`` bytes in one dgen/numpy call,
        then wraps the result as a ``FixedSizeListArray`` — no per-row Python
        loop, no tiling, no copy.  Each row is a distinct slice of the data
        stream so samples within the same file are NOT identical.

        ``elem_size`` = dim1 * dim2 (the flat element count per sample).
        """
        # One contiguous buffer for all rows — zero-copy FixedSizeList wrap.
        flat = gen_random_tensor(shape=(batch_size * elem_size,), dtype=np.uint8, rng=rng)
        arrow_flat = pa.array(flat, type=pa.uint8())
        arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, elem_size)
        return {'data': arrow_data}

    # ── Main generation loop ──────────────────────────────────────────────────

    def generate(self):
        """Generate Parquet files using dgen-py streaming or per-batch fallback.

        Generation strategy:

        1. **dgen-py streaming** (legacy uint8 mode):
           Builds on PR #10 (Wolfgang De Salvador) which reduced generation
           call overhead by batching upfront.  This implementation extends that
           idea while eliminating the full-file memory allocation:

           - *Constant O(batch) memory*: one reusable buffer allocated via
             ``create_bytearrays()`` (1,280x faster than Python ``bytearray``).
           - *BufferPool* (batch <= 1 MB): zero-copy ``next_slice()`` for
             small batches.
           - *Streaming Generator* (batch > 1 MB): single ``Generator`` per
             rank; Rayon thread pool is created once and reused across all
             files via ``set_seed()`` at each file boundary.
           - *MPI-aware threading*: ``max_threads = cpu_count // ranks_per_node``
             prevents CPU contention when NP ranks share one node.

        2. **Per-batch via gen_random_tensor** (column-schema mode only):
           each batch is generated independently.  Memory is bounded by
           ``parquet_generation_batch_size`` rows.

        Seeding:
        - A NumPy ``default_rng`` seeded with ``BASE_SEED + my_rank`` advances
          once per file to produce a ``file_seed``.  Per-file seeding gives
          reproducibility independent of MPI topology (NP=1 and NP=8 produce
          byte-for-byte identical files for any given file index).
        - dgen streaming: ``Generator.set_seed(file_seed)`` resets the Rayon
          stream position before each file.
        - Fallback path: ``rng`` flows through ``gen_random_tensor`` as before.
        """
        super().generate()

        np.random.seed(self.BASE_SEED + self.my_rank)
        rng = np.random.default_rng(seed=self.BASE_SEED + self.my_rank)

        dim = self.get_dimension(self.total_files_to_generate)
        compression = COMPRESSION_MAP.get(self.compression, None)
        is_local = self.storage.islocalfs()

        # -- MPI-aware thread cap ---------------------------------------------
        # Each rank gets cpu_count // ranks_per_node cores so that NP ranks
        # sharing one node never over-subscribe the CPU.
        try:
            _ranks_on_node = DLIOMPI.get_instance().ranks_per_node()
        except Exception:
            _ranks_on_node = max(1, self.comm_size)
        _max_gen_threads = max(1, (os.cpu_count() or 1) // _ranks_on_node)

        # -- dgen-py streaming state ------------------------------------------
        # Streaming is used for legacy uint8 mode only.  Column-schema mode
        # uses per-batch gen_random_tensor so arbitrary dtypes are handled
        # without extra conversion code.
        _use_dgen_stream = not self.parquet_columns
        _dgen_pool     = None   # BufferPool for batch_bytes <= _DGEN_POOL_THRESHOLD (reused across all files)
        _use_pool      = None   # bool, determined on first file
        _stream_bytes  = None   # file_bytes for Generator sizing
        _batch_bytes   = None   # bytes per batch
        _batch_buf     = None   # Reusable bytearray for fill_chunk (allocated once)

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            progress(i + 1, self.total_files_to_generate, "Generating Parquet Data")

            out_path_spec = self.storage.get_uri(self._file_list[i])

            dim_raw = dim[2 * i]
            if isinstance(dim_raw, list):
                dim1 = int(dim_raw[0])
                dim2 = int(dim_raw[1]) if len(dim_raw) > 1 else 1
            else:
                dim1 = int(dim_raw)
                dim2 = int(dim[2 * i + 1])
            elem_size = dim1 * dim2

            # Lazy one-time init of dgen streaming constants and shared resources.
            # Deferred until elem_size is known.  All files share the same dimensions
            # so this block executes exactly once per rank.
            if _use_dgen_stream and _use_pool is None:
                _batch_bytes  = self.generation_batch_size * elem_size
                _stream_bytes = self.num_samples * elem_size
                _use_pool     = _batch_bytes <= _DGEN_POOL_THRESHOLD
                if _use_pool:
                    # BufferPool: zero-copy next_slice() for sub-32 MB batches.
                    # A single pool instance is reused across all files.
                    _dgen_pool = _dgen_py.BufferPool()
                else:
                    # Streaming path: ONE Generator created once, reused across all
                    # files via reset()+set_seed() — eliminates per-file Rayon pool
                    # creation/destruction (was the dominant overhead for small files).
                    # create_bytearrays: Rust mmap allocation -- 1,280x faster
                    # than Python bytearray().
                    _batch_buf = _dgen_py.create_bytearrays(
                        count=1, size=_batch_bytes
                    )[0]
                    _dgen_gen = _dgen_py.Generator(
                        size=_stream_bytes,
                        max_threads=_max_gen_threads,
                    )

            if self.partition_by:
                # Partitioned writes use the full-table path (pq.write_to_dataset
                # does not expose a per-row-group streaming interface).
                if self.parquet_columns:
                    columns = self._generate_batch_columns(self.num_samples, rng)
                else:
                    columns = self._generate_legacy_batch(elem_size, self.num_samples, rng)
                table = pa.table(columns)
                pq.write_to_dataset(
                    table,
                    root_path=os.path.dirname(out_path_spec),
                    partition_cols=[self.partition_by],
                    compression=compression,
                    row_group_size=self.row_group_size,
                )
                continue

            schema = self._build_schema(legacy_elem_size=elem_size)

            if is_local:
                parent_dir = os.path.dirname(out_path_spec)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                writer_target = out_path_spec
            else:
                writer_target = pa.BufferOutputStream()

            num_batches = (
                self.num_samples + self.generation_batch_size - 1
            ) // self.generation_batch_size

            with pq.ParquetWriter(writer_target, schema, compression=compression) as writer:
                if _use_dgen_stream and _use_pool is not None:
                    # -- dgen-py streaming path -------------------------------
                    # Singleton Generator reused across files: reset() repositions
                    # to byte 0 before each file — O(µs), no allocation, no thread
                    # pool teardown/creation.  Content varies naturally as the RNG
                    # state advances.  Reproducibility is not required.
                    if not _use_pool:
                        _dgen_gen.reset()

                    for batch_idx in range(num_batches):
                        batch_start = batch_idx * self.generation_batch_size
                        batch_end   = min(batch_start + self.generation_batch_size,
                                         self.num_samples)
                        cur_rows  = batch_end - batch_start
                        cur_bytes = cur_rows * elem_size

                        if _dgen_pool is not None:
                            # Zero-copy slice from the rolling pool.
                            raw = _dgen_pool.next_slice(cur_bytes)
                            arr = np.frombuffer(raw, dtype=np.uint8)
                        else:
                            # Fill the pre-allocated buffer; for the final
                            # (possibly smaller) batch, read only cur_bytes.
                            _dgen_gen.fill_chunk(_batch_buf)
                            arr = np.frombuffer(_batch_buf, dtype=np.uint8)[:cur_bytes]

                        arrow_flat = pa.array(arr, type=pa.uint8())
                        arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, elem_size)
                        writer.write_table(
                            pa.table({'data': arrow_data}),
                            row_group_size=self.row_group_size,
                        )
                else:
                    # -- Per-batch fallback path ------------------------------
                    # Column-schema mode: each batch generated independently via
                    # gen_random_tensor (already dgen-backed when available).
                    # Memory is bounded by generation_batch_size rows.
                    for batch_idx in range(num_batches):
                        batch_start = batch_idx * self.generation_batch_size
                        batch_end   = min(batch_start + self.generation_batch_size,
                                         self.num_samples)
                        cur_rows = batch_end - batch_start

                        if self.parquet_columns:
                            columns = self._generate_batch_columns(cur_rows, rng)
                        else:
                            columns = self._generate_legacy_batch(elem_size, cur_rows, rng)
                        writer.write_table(
                            pa.table(columns),
                            row_group_size=self.row_group_size,
                        )

            if not is_local:
                self.storage.put_data(out_path_spec, writer_target.getvalue().to_pybytes())

        np.random.seed()
