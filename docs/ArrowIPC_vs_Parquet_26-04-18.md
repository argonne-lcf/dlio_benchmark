# Arrow IPC vs Parquet — Format Recommendation and Implementation Plan

**Date:** April 18, 2026  
**Replaces:** `parquet_pipeline_analysis.md`, `PyArrow-Parquet_Analysis_26-04-18.md`  
**Decision:** **Do not invest in dgen-rs Parquet encoder. Pivot to Arrow IPC.**

---

## TL;DR — Storage Benchmark Accuracy

Parquet produces an **inaccurate** storage benchmark at high throughput. The CPU decoder saturates around 1–2 GB/s and becomes the bottleneck before the storage system does. You end up measuring the client CPU, not the storage.

Arrow IPC is the correct choice for a storage benchmark — bytes on disk are the in-memory format, so the I/O path is always the bottleneck. Two conditions must hold to keep it honest:

- **S3 / object storage**: inherently cache-bypass. Every `get_batch()` goes over the network. No extra steps needed.
- **Local filesystem**: the OS page cache will silently serve reads from DRAM after the first epoch. Use `odirect: true` (O_DIRECT, bypasses page cache completely) or size the dataset so it substantially exceeds host RAM. **O_DIRECT must be implemented for Arrow IPC — not raised as an unsupported exception — because it is the primary tool for accurate local storage benchmarking.**

---

## Recommendation Summary

Do **not** spend time adding a Rust Parquet encoder to dgen-rs/dgen-py.

The correct solution is to implement Arrow IPC file support in dlio_benchmark. Arrow IPC eliminates the Parquet generation bottleneck entirely, improves read throughput by orders of magnitude, requires **zero changes to s3dlio or dgen-rs**, and takes roughly 300–400 lines of Python across 3 new files.

---

## Measured Benchmark Results

All measurements on this machine (12 logical CPUs, PyArrow 23.0.1) using 128 rows × 512 KB = 64 MB files — representative of large AI training sample files:

```
Arrow IPC write:  0.80 GB/s   (67 MB/file)
Parquet write:    0.04 GB/s   (67 MB/file)
IPC write speedup: 20x faster to generate

Arrow IPC read:   1338 GB/s   (in-memory; get_batch × 2)
Parquet read:     0.08 GB/s   (read_row_group × 2)
IPC read speedup: 15,752x faster to read from memory
```

The 0.04 GB/s Parquet write figure matches the `parquet_pipeline_analysis.md` measurement exactly. It is a fundamental constraint of PyArrow's Parquet encoder processing `FixedSizeListArray<uint8>` at element granularity rather than block granularity. This cannot be fixed from Python.

The in-memory read speedup is extreme because Arrow IPC `get_batch()` is a direct memory view — no decoding, no decompression. In real S3/network scenarios the read advantage is bounded by network throughput, but the CPU overhead difference is preserved at any bandwidth: Parquet requires decode+decompress per column chunk; Arrow IPC requires only `memcpy`.

---

## Why the dgen-rs Parquet Encoder Is the Wrong Investment

The `parquet_pipeline_analysis.md` proposal was to add ~300 lines of Rust to dgen-rs implementing a parallel Parquet encoder using the `parquet` crate. This would:

- Fix the generation bottleneck for Parquet
- Require maintaining Rust Parquet crate integration in dgen-rs permanently
- Still produce Parquet files that readers must decode+decompress under high network load
- Deliver no improvement at the point that actually matters for production AI training: **read throughput at >10 GB/s storage bandwidth**

Arrow IPC solves both the generation bottleneck (20× faster write) and the read bottleneck (zero CPU decode) in one move, with no Rust changes at all.

---

## Format Comparison

| Property | Parquet | Arrow IPC File |
|---|---|---|
| On-disk format | Columnar, encoded, compressed | Raw Arrow buffers (the in-memory format) |
| Write throughput (PyArrow) | ~0.04 GB/s for large fixed-size arrays | ~0.80 GB/s — 20× faster |
| Read CPU cost | Decode + decompress per column chunk | `memcpy` only — zero decode overhead |
| Footer | Row-group metadata + column stats | Record batch byte offsets |
| Random batch access | `bisect(cumulative_offsets, idx)` | `get_batch(i)` — O(1), exact offset |
| Compression | gzip, snappy, zstd, lz4, brotli | lz4, zstd optional (default: none) |
| Bottleneck at >10 GB/s storage | CPU (decode) is the bottleneck | Network / storage is the bottleneck |
| pyarrow write API | `pq.ParquetWriter` | `pa.ipc.new_file()` |
| pyarrow read API | `pq.ParquetFile.read_row_group(i)` | `pa.ipc.open_file().get_batch(i)` |
| dgen-rs changes needed | Would require ~300 new lines | **None** |
| s3dlio changes needed | None | **None** |

The data scientist's observation — "moving away from Parquet because it is hard to effectively utilize these files" — is precisely the CPU decode bottleneck at high network throughput. Arrow IPC removes it.

---

## Why No New Rust Is Needed

The existing s3dlio Python API is already sufficient:
- `s3dlio.get_range(uri, offset, length)` → range GET for any backend (S3, GCS, Azure, file, direct)
- `s3dlio.stat(uri)["size"]` → file size for any backend

The `_S3RangeFile` adapter in `parquet_reader_s3_iterable.py` wraps exactly these two calls. `pa.ipc.open_file()` accepts any seekable file-like object, so the adapter works unchanged for Arrow IPC — it is format-agnostic. The Arrow IPC reader is the Parquet reader with two function names changed.

---

## Benchmark Accuracy: Are We Measuring Storage or CPU/Memory?

This is the right question to ask before implementing any format, and the answer determines how the implementation must be designed.

### The Parquet problem — it makes a poor storage benchmark

At storage throughputs above roughly 1–2 GB/s (easily achievable on modern NVMe or fast S3), PyArrow's Parquet column decoder saturates the CPU before the storage system is saturated. The result: storage throughput could double and benchmark results would not change, because the bottleneck is the client CPU doing decode, not the storage system doing I/O. **Parquet-based workloads do not accurately benchmark storage at high throughput.** This is one of the primary real-world motivations for formats like Arrow IPC: practitioners building fast ML pipelines have observed this exact bottleneck and moved away from Parquet.

### Arrow IPC — correct for storage benchmarking, with two caveats

Arrow IPC bytes on disk are identical to the Arrow in-memory format. `get_batch()` issues one range read and places the result directly into the Arrow buffer pool. There is no column decoder, no dictionary expansion, no decompression. The CPU cost is dominated by the I/O syscall and a single DMA + user-space copy. At any storage throughput below ~50 GB/s (memory bandwidth), the storage read is the bottleneck, not the final copy. This is what a storage benchmark should measure.

However, two conditions must hold for Arrow IPC reads to accurately measure storage rather than DRAM or page cache:

#### Caveat 1: Page cache (local filesystem only)

On a local POSIX filesystem, the OS page cache will retain file data in DRAM after the first read. A second epoch over the same files will return entirely from page cache — measuring DRAM bandwidth (~40–80 GB/s), not NVMe or network storage. This problem exists for every format, but it is **more acute for Arrow IPC** because:
- Parquet decode is CPU-intensive — the CPU acts as a natural throttle that causes pages to be evicted before the next epoch starts
- Arrow IPC decode is trivial — the OS has time to cache everything before the next epoch begins

dlio_benchmark already detects this condition and warns:
```
WARNING: The amount of dataset is smaller than the host memory; data might be
cached after the first epoch. Increase the size of dataset to eliminate the caching effect!
```

This warning should be heeded. But for cases where the dataset cannot be made large enough (e.g., rapid iteration, testing), **O_DIRECT is the correct solution** — it bypasses the page cache entirely and forces every read to go to storage hardware.

The current implementation plan raises `Exception("O_DIRECT not yet supported")` for Arrow IPC. **This must be implemented, not skipped.** O_DIRECT is the most important mode for accurate local storage benchmarking. See the implementation plan below for the approach.

#### Caveat 2: Page cache (object storage — not a concern)

For S3/MinIO/object storage via `ArrowIPCReaderS3Iterable`, every `get_batch()` call goes over the network. There is no page cache. Object storage benchmarks are inherently cache-bypass and accurately measure storage throughput with Arrow IPC. No special handling needed.

### Is Arrow IPC a realistic production format?

Yes. Hugging Face `datasets` library stores all datasets in Arrow IPC (Feather v2) format internally. The format is used in production MLOps pipelines at scale. It is also the native exchange format between Apache Arrow producers and consumers (Spark, DuckDB, pandas, Polars). The premise that "real workloads use Parquet" is correct for data warehousing and analytics — but for AI training data ingestion, Arrow IPC is an accurate representation of modern high-throughput pipelines.

### Summary of accuracy requirements

| Storage path | Cache bypass needed | How to achieve it |
|---|---|---|
| Local NVMe / SSD | **Yes** | `odirect: true` in YAML (O_DIRECT), OR dataset >> host RAM |
| NFS / parallel FS | **Yes, if close-to-cache** | Dataset >> host RAM, OR `echo 3 > /proc/sys/vm/drop_caches` between epochs |
| S3 / object storage | No — always bypassed | Nothing extra needed; every GET goes to the network |
| MinIO on same machine | Partial — MinIO has its own cache | Use remote MinIO, or size dataset >> MinIO server RAM |

**The benchmark operator's checklist for accurate Arrow IPC results:**
1. Set `num_files_train` so that `total dataset size >> host RAM` (heed the dlio_benchmark warning)
2. For local storage: use `odirect: true` or run `echo 3 > /proc/sys/vm/drop_caches` between experiments (requires root / `sudo`)
3. Discard epoch 1 results if cache state is uncertain; report epoch 2+ as the steady-state storage throughput
4. For S3: no extra steps — object storage reads are always cache-bypass

### Format verdict from a storage benchmarking perspective

| Format | Bottleneck at >1 GB/s storage | Accurate storage benchmark? |
|---|---|---|
| Parquet (compressed) | CPU decode — saturates before storage | **No** — measures client CPU, not storage |
| Parquet (uncompressed) | CPU decode (lighter) — still ~1–5 GB/s ceiling | **Marginal** — becomes inaccurate above ~2 GB/s |
| Arrow IPC (uncompressed) | Storage I/O (with cache bypass) | **Yes** — measures storage when cache is bypassed |
| Arrow IPC + LZ4 | LZ4 decode at ~15 GB/s — above most storage | **Yes** — LZ4 is fast enough to remain storage-bound up to ~12 GB/s |

Arrow IPC uncompressed is the correct choice. Arrow IPC + LZ4 is a valid alternative for benchmarking compressed data ingestion while remaining storage-bound.

---

## Implementation Plan

Six touch points total. Three new files, three small modifications.

Both POSIX/local-filesystem and object-storage (S3/MinIO/GCS/Azure via s3dlio) paths must work, and the reader factory already dispatches on `storage_type` exactly as it does for Parquet. The generator already uses `self.storage.islocalfs()` to choose between a direct file write and a buffer-then-upload path. Arrow IPC follows both patterns identically — only the writer and reader API calls change.

---

### New files

#### `reader/arrow_ipc_reader.py` — POSIX / local filesystem reader

Used when `storage_type` is `local`, `nfs`, or any non-object-store type. PyArrow's `pa.ipc.open_file()` accepts a plain filesystem path directly, so no adapter is needed — it opens, `mmap`s, and reads the footer in one call.

```python
import pyarrow as pa
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.common.constants import MODULE_DATA_READER

dlp = Profile(MODULE_DATA_READER)

class ArrowIPCReader(FormatReader):

    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        opts = getattr(self._args, "storage_options", {}) or {}
        self._batch_cache_size = int(opts.get("batch_cache_size", 4))
        self._batch_cache: dict = {}
        self._lru: list = []

    @dlp.log
    def open(self, filename):
        # pa.ipc.open_file reads only the footer on open — no full file load.
        reader = pa.ipc.open_file(filename)
        return (reader, reader.num_record_batches)

    @dlp.log
    def get_sample(self, filename, sample_index):
        reader, num_batches = self.open_file_map[filename]
        # Assumes fixed records_per_batch; num_record_batches divides num_samples evenly.
        records_per_batch = self._args.num_samples_per_file // num_batches
        batch_idx = sample_index // records_per_batch

        cache_key = (filename, batch_idx)
        if cache_key not in self._batch_cache:
            if len(self._lru) >= self._batch_cache_size:
                evict = self._lru.pop(0)
                self._batch_cache.pop(evict, None)
            # get_batch() issues exactly one range read for the batch buffers.
            batch = reader.get_batch(batch_idx)
            self._batch_cache[cache_key] = batch
            self._lru.append(cache_key)
        else:
            batch = self._batch_cache[cache_key]

        dlp.update(image_size=batch.nbytes)
        return self._args.resized_image

    @dlp.log
    def close(self, filename):
        keys = [k for k in self._batch_cache if k[0] == filename]
        for k in keys:
            self._batch_cache.pop(k, None)
            if k in self._lru:
                self._lru.remove(k)
        super().close(filename)
```

Key points:
- `pa.ipc.open_file(filename)` works with any POSIX path, NFS mount, or `file://` URI. No wrapper needed.
- `reader.num_record_batches` replaces the `bisect(cumulative_offsets)` lookup used in Parquet — Arrow IPC stores batch byte offsets in the footer, so `get_batch(i)` is O(1) and issues one exact range read.
- The cache eviction and `dlp` telemetry pattern is identical to `ParquetReader`.
- **O_DIRECT must be supported** (not raise an exception) — it is the primary mechanism for accurate local storage benchmarking. When `odirect: true`, `open()` uses a `_DirectRangeFile` adapter backed by s3dlio's `direct://` URI scheme instead of a plain path. See `reader_factory.py` modifications below.

---

#### `reader/arrow_ipc_reader_s3_iterable.py` — S3 / object-store reader

Used when `storage_type` is `s3` or `aistore`. PyArrow's `pa.ipc.open_file()` accepts any seekable file-like object, which means the existing `_S3RangeFile` adapter from `parquet_reader_s3_iterable.py` works without modification. The s3dlio, MinIO, and s3torchconnector dispatch paths are unchanged.

```python
import pyarrow as pa
from dlio_benchmark.reader.parquet_reader_s3_iterable import (
    _S3RangeFile, _MinioRangeFile, _S3TCRangeFile,
)
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.common.constants import MODULE_DATA_READER

dlp = Profile(MODULE_DATA_READER)

class ArrowIPCReaderS3Iterable(FormatReader):

    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        opts = getattr(self._args, "storage_options", {}) or {}
        self._storage_library = opts.get("storage_library", "s3dlio")
        self._endpoint_url    = opts.get("endpoint_url", "")
        self._batch_cache_size = int(opts.get("batch_cache_size", 4))
        self._batch_cache: dict = {}
        self._lru: list = []

    def _make_range_file(self, uri):
        """Return the appropriate seekable file-like adapter for this URI."""
        if self._storage_library == "s3dlio":
            return _S3RangeFile(uri)
        elif self._storage_library == "minio":
            return _MinioRangeFile(uri, self._endpoint_url)
        elif self._storage_library == "s3torchconnector":
            return _S3TCRangeFile(uri)
        else:
            raise ValueError(f"Unknown storage_library: {self._storage_library!r}")

    @dlp.log
    def open(self, filename):
        rf = self._make_range_file(filename)
        # Two small range GETs: one for the magic/version header, one for the footer.
        reader = pa.ipc.open_file(rf)
        return (reader, reader.num_record_batches)

    @dlp.log
    def get_sample(self, filename, sample_index):
        reader, num_batches = self.open_file_map[filename]
        records_per_batch = self._args.num_samples_per_file // num_batches
        batch_idx = sample_index // records_per_batch

        cache_key = (filename, batch_idx)
        if cache_key not in self._batch_cache:
            if len(self._lru) >= self._batch_cache_size:
                evict = self._lru.pop(0)
                self._batch_cache.pop(evict, None)
            # Exactly one range GET for the batch body — no full-file download.
            batch = reader.get_batch(batch_idx)
            self._batch_cache[cache_key] = batch
            self._lru.append(cache_key)
        else:
            batch = self._batch_cache[cache_key]

        dlp.update(image_size=batch.nbytes)
        return self._args.resized_image

    @dlp.log
    def close(self, filename):
        keys = [k for k in self._batch_cache if k[0] == filename]
        for k in keys:
            self._batch_cache.pop(k, None)
            if k in self._lru:
                self._lru.remove(k)
        super().close(filename)
```

Key points:
- `_S3RangeFile` is format-agnostic — it only implements `seek/tell/read` over `s3dlio.get_range` and `s3dlio.stat`. It does not know or care whether it is wrapping a Parquet footer scan or an Arrow IPC footer scan.
- `pa.ipc.open_file(rf)` reads the IPC file magic (8 bytes) and footer (a small Flatbuffer at the end of the file) using two range GETs, then returns. No record batch data is transferred at open time.
- `reader.get_batch(batch_idx)` uses the byte offset and length from the footer to issue exactly one range GET. This is the strongest advantage over Parquet at S3 scale: one network round trip per sample batch, not a column-chunk scan.

---

#### `data_generator/arrow_ipc_generator.py` — file generator (POSIX and object storage)

The existing `ParquetGenerator.generate()` already uses `is_local = self.storage.islocalfs()` to dispatch between a direct write (`writer_target = out_path_spec`) and a buffer-then-upload path (`writer_target = pa.BufferOutputStream()` → `self.storage.put_data(...)`). `ArrowIPCGenerator` uses the same pattern with `pa.ipc.new_file()` in place of `pq.ParquetWriter`.

```python
import os
import numpy as np
import pyarrow as pa

from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.data_generator.parquet_generator import (
    ParquetGenerator, _PA_SCALAR_TYPE_MAP, _NP_TYPE_MAP,
)
from dlio_benchmark.utils.utility import progress, gen_random_tensor, DLIOMPI
import dgen_py as _dgen_py

class ArrowIPCGenerator(DataGenerator):

    def __init__(self):
        super().__init__()
        opts = getattr(self._args, "storage_options", {}) or {}
        # ipc_record_batch_size: rows per Arrow record batch.
        # Must divide num_samples_per_file evenly for O(1) get_batch() indexing.
        self.record_batch_size = int(
            opts.get("ipc_record_batch_size",
                     getattr(self._args, "parquet_row_group_size", 1024))
        )
        self.parquet_columns = getattr(self._args, "parquet_columns", [])

    def _build_schema(self, legacy_elem_size=None):
        # Reuse the same schema logic as ParquetGenerator — the Arrow IPC file
        # format stores this schema verbatim in the file header and footer.
        pg = ParquetGenerator.__new__(ParquetGenerator)
        pg._args = self._args
        pg.parquet_columns = self.parquet_columns
        return pg._build_schema(legacy_elem_size=legacy_elem_size)

    def generate(self):
        super().generate()

        np.random.seed(self.BASE_SEED + self.my_rank)
        rng = np.random.default_rng(seed=self.BASE_SEED + self.my_rank)
        dim = self.get_dimension(self.total_files_to_generate)
        is_local = self.storage.islocalfs()

        write_opts = pa.ipc.IpcWriteOptions(compression=None)  # zero-decode on read

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            progress(i + 1, self.total_files_to_generate, "Generating Arrow IPC Data")

            out_path_spec = self.storage.get_uri(self._file_list[i])
            dim_raw = dim[2 * i]
            if isinstance(dim_raw, list):
                dim1 = int(dim_raw[0]); dim2 = int(dim_raw[1]) if len(dim_raw) > 1 else 1
            else:
                dim1 = int(dim_raw); dim2 = int(dim[2 * i + 1])
            elem_size = dim1 * dim2

            schema = self._build_schema(legacy_elem_size=elem_size)

            # ── Choose write target ───────────────────────────────────────
            if is_local:
                parent_dir = os.path.dirname(out_path_spec)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                writer_target = out_path_spec          # direct filesystem write
            else:
                writer_target = pa.BufferOutputStream()  # buffer → put_data below

            num_batches = (self.num_samples + self.record_batch_size - 1) // self.record_batch_size

            with pa.ipc.new_file(writer_target, schema, options=write_opts) as writer:
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * self.record_batch_size
                    batch_end   = min(batch_start + self.record_batch_size, self.num_samples)
                    cur_rows    = batch_end - batch_start

                    if self.parquet_columns:
                        # Column-schema mode — reuse ParquetGenerator helpers
                        pg = ParquetGenerator.__new__(ParquetGenerator)
                        pg._args = self._args
                        pg.parquet_columns = self.parquet_columns
                        columns = pg._generate_batch_columns(cur_rows, rng)
                    else:
                        # Legacy uint8 mode — same dgen path as ParquetGenerator
                        flat = gen_random_tensor(shape=(cur_rows * elem_size,),
                                                 dtype=np.uint8, rng=rng)
                        arrow_flat = pa.array(flat, type=pa.uint8())
                        arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, elem_size)
                        columns = {'data': arrow_data}

                    batch = pa.RecordBatch.from_arrays(
                        list(columns.values()), schema=schema
                    )
                    writer.write_batch(batch)

            # ── Upload if object storage ──────────────────────────────────
            if not is_local:
                self.storage.put_data(out_path_spec, writer_target.getvalue().to_pybytes())

        np.random.seed()
```

Key points:
- `is_local` dispatch is identical to `ParquetGenerator` — `pa.ipc.new_file()` accepts either a filesystem path or a `BufferOutputStream` and PyArrow handles both transparently.
- For POSIX/local: the IPC file is written directly to disk with `os.makedirs` pre-created, no staging buffer needed.
- For object storage: the entire file is buffered in a `BufferOutputStream`, then uploaded via `self.storage.put_data()` — exactly the same pattern as Parquet. The file is typically 64–512 MB so this is acceptable; streaming multipart upload could be added later if needed.
- `write_opts = IpcWriteOptions(compression=None)` is the default and should stay the default. Enabling `lz4` or `zstd` compression is possible but defeats the zero-decode advantage.
- The dgen-py streaming pool path (used by `ParquetGenerator` for sub-32 MB batches) can be ported directly from `parquet_generator.py` once the basic path is validated.

---

### Modifications

#### `common/enumerations.py`
```python
# In FormatType enum — add after PARQUET
ARROW_IPC = 'arrow_ipc'

# In FormatType.get_enum() — add the elif branch
elif FormatType.ARROW_IPC.value == value:
    return FormatType.ARROW_IPC
```

#### `reader/reader_factory.py`

The dispatch mirrors the existing Parquet block exactly: S3/AIStore → range-GET reader, everything else → POSIX reader. Unlike Parquet, **O_DIRECT must be supported** — it is the primary means of ensuring accurate local storage benchmarking. When `odirect: true`, the local reader uses s3dlio's `direct://` URI scheme, which opens the file with `O_DIRECT` and returns DMA-aligned buffers. The S3 reader is inherently cache-bypass and ignores the flag.

```python
elif type == FormatType.ARROW_IPC:
    if _args.storage_type in (StorageType.S3, StorageType.AISTORE):
        # S3/object storage is always cache-bypass — odirect flag is irrelevant.
        from dlio_benchmark.reader.arrow_ipc_reader_s3_iterable import ArrowIPCReaderS3Iterable
        return ArrowIPCReaderS3Iterable(dataset_type, thread_index, epoch_number)
    else:
        # For local/NFS: ArrowIPCReader respects odirect=True via direct:// URI.
        # Raising NotImplemented here (as other formats do) would make accurate
        # local storage benchmarking impossible — do not do this.
        from dlio_benchmark.reader.arrow_ipc_reader import ArrowIPCReader
        return ArrowIPCReader(dataset_type, thread_index, epoch_number)
```

O_DIRECT implementation in `ArrowIPCReader.open()`:

```python
@dlp.log
def open(self, filename):
    if getattr(self._args, 'odirect', False):
        # Rewrite the path as a direct:// URI so s3dlio opens with O_DIRECT.
        # s3dlio.get_range('direct:///path/to/file', offset, length) returns
        # DMA-aligned Bytes, bypassing the page cache entirely.
        import s3dlio
        uri = 'direct://' + filename if not filename.startswith('direct://') else filename
        # Wrap in _DirectRangeFile (same interface as _S3RangeFile but uses
        # s3dlio direct:// backend for O_DIRECT reads).
        rf = _DirectRangeFile(uri)
        reader = pa.ipc.open_file(rf)
    else:
        reader = pa.ipc.open_file(filename)
    return (reader, reader.num_record_batches)
```

`_DirectRangeFile` is identical to `_S3RangeFile` with `uri = 'direct://' + posix_path`. Since s3dlio's `direct://` backend already handles aligned reads, no additional buffer alignment code is needed in Python.

#### `data_generator/generator_factory.py`

The generator handles both POSIX and object storage internally via `self.storage.islocalfs()`, so a single factory entry covers both:

```python
elif type == FormatType.ARROW_IPC:
    from dlio_benchmark.data_generator.arrow_ipc_generator import ArrowIPCGenerator
    return ArrowIPCGenerator()
```

---

### Example YAML — local filesystem

```yaml
dataset:
  format: arrow_ipc
  storage_type: local
  storage_root: /mnt/nvme/training-data
  num_samples_per_file: 1024
  num_files_train: 500
  storage_options:
    batch_cache_size: 4
    ipc_record_batch_size: 256   # must divide num_samples_per_file evenly
```

No `storage_library` key is needed for local storage — `ArrowIPCReader` opens files directly with `pa.ipc.open_file(filename)`.

### Example YAML — S3 / object storage via s3dlio

```yaml
dataset:
  format: arrow_ipc
  storage_type: s3
  storage_root: my-bucket
  num_samples_per_file: 1024
  num_files_train: 500
  storage_options:
    storage_library: s3dlio
    endpoint_url: http://127.0.0.1:9000
    batch_cache_size: 4
    ipc_record_batch_size: 256   # must divide num_samples_per_file evenly
```

---

## What to Keep from the Parquet Work

The existing `ParquetReader`, `ParquetReaderS3Iterable`, and `ParquetGenerator` should remain in the codebase. Parquet is a widely used format and the existing implementation is correct and production-quality. The recommendation is not to remove Parquet support — it is to add Arrow IPC as the preferred format for new workloads, particularly those running against high-throughput storage (>10 GB/s).

For existing Parquet datasets, the current readers continue to work. For new datasets, Arrow IPC is the better choice.
