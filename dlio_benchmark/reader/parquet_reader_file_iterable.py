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
"""
Parquet reader for local and network filesystems using raw byte-range reads.

Designed to be structurally identical to ParquetReaderS3Iterable so that
file-storage and object-storage benchmarks are directly comparable:

  S3 path   : HTTP byte-range GET  →  compressed RG bytes arrive over network
  File path : open()/seek()/read() →  compressed RG bytes arrive from storage

Both paths:
  • Read ONLY the compressed row-group bytes (no pyarrow decode/decompress).
  • Discard the data immediately — this is a storage benchmark.
  • Report the same compressed-byte count to dlp telemetry.
  • Use a ThreadPoolExecutor with `prefetch_workers` threads per worker process.
  • Expose open_footer_only() / submit_rg_prefetch() for the sliding-window
    iterator in TorchIterableDataset — the identical API means the same
    __iter__ code path drives both backends.

The key parity decision: we do NOT call pf.read_row_group() (which decompresses
the data in Python) because the S3 path never decompresses. We compute the exact
byte offset and length of each row group's column chunks from the parquet footer
metadata, then issue a raw open/seek/read — matching the S3 range GET semantics
byte-for-byte.

Configuration (under storage_options in the DLIO YAML):
  prefetch_workers: 64   # threads per worker process (default 64)
  columns:          null # list of column names (null = all)
  prefetch_window:  64   # sliding-window depth (controlled by TorchIterableDataset)

Example YAML snippet:
  storage:
    storage_type: local_fs
    storage_root: /path/to/parquet/root
  dataset:
    format: parquet
    num_samples_per_file: 1000000
    storage_options:
      prefetch_workers: 64
"""
import bisect
import os
from concurrent.futures import ThreadPoolExecutor

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import ReadType as _ReadType
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ParquetReaderFileIterable(FormatReader):
    """
    Row-group-granular Parquet reader for local/network filesystems.

    Reads compressed row-group bytes via open()/seek()/read() in a thread pool,
    matching the throughput measurement semantics of ParquetReaderS3Iterable.

    Exposes the same sliding-window API (open_footer_only / submit_rg_prefetch)
    so that TorchIterableDataset drives both S3 and file backends identically.

    DLIO's FormatReader protocol:
      open(filename)            → returns (ParquetFile, None, cumulative_offsets)
      get_sample(filename, idx) → bisect-locates RG, waits on prefetch future,
                                  records compressed bytes, updates dlp
      close(filename)           → no-op (caches kept for full epoch)
      next() / read_index()     → delegate to FormatReader base class

    Cache format: _pf_cache[filename] = (pf, None, offsets)
      The second element is None (no file-handle adapter needed; raw reads
      open their own fd per call for thread safety).
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        opts = getattr(self._args, "storage_options", {}) or {}

        # Optional column selection (list[str] or None = all columns)
        self._columns = opts.get("columns") or None

        # Footer cache: filename → (ParquetFile, None, cumulative_offsets).
        # Holds ONLY the Parquet footer metadata, a few KB per file.
        # Flushed at finalize() (epoch boundary).
        self._pf_cache: dict = {}

        # Row-group byte-count cache: (filename, rg_idx) → int (compressed bytes).
        # Flushed at finalize().
        self._rg_cache: dict = {}

        # Prefetch futures: (filename, rg_idx) → Future[int]
        self._prefetch_futures: dict = {}

        # Thread pool for parallel raw byte-range reads.
        max_w = int(opts.get("prefetch_workers", 64))
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=max_w, thread_name_prefix="rg-file-prefetch"
        )

        self.logger.info(
            f"{utcnow()} ParquetReaderFileIterable "
            f"thread={thread_index} epoch={epoch} "
            f"prefetch_workers={max_w} columns={self._columns}"
        )

    # ── I/O helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_rg_file(filename: str, rg_start: int, rg_len: int, compressed_bytes: int) -> int:
        """Read one row-group's compressed bytes from disk in a background thread.

        Opens a private file descriptor (thread-safe), seeks to rg_start, reads
        rg_len bytes, discards them immediately.  Returns compressed_bytes so
        the Future carries the value needed by get_sample() without extra state.

        This is the file-storage analogue of _fetch_rg_s3torch() in
        ParquetReaderS3Iterable: both read exactly the compressed column bytes
        and discard them — no decode, no pyarrow Table, no memory held.
        """
        if rg_len > 0:
            with open(filename, "rb") as fh:
                fh.seek(rg_start)
                fh.read(rg_len)
        return compressed_bytes

    def _rg_byte_range(self, pf, rg_idx):
        """Compute (rg_start_bytes, rg_len_bytes, all_compressed_bytes) for rg_idx.

        Uses the same column-chunk offset arithmetic as ParquetReaderS3Iterable
        so that both backends measure identical byte counts.
        """
        meta = pf.metadata
        schema = pf.schema_arrow
        n_cols = meta.row_group(0).num_columns if meta.num_row_groups > 0 else 0

        if self._columns is not None:
            col_names = set(self._columns)
            col_indices = [i for i in range(n_cols) if schema.field(i).name in col_names]
        else:
            col_indices = list(range(n_cols))

        rg_meta = meta.row_group(rg_idx)
        all_comp = sum(
            rg_meta.column(c).total_compressed_size for c in range(rg_meta.num_columns)
        )
        rg_start_b = rg_end_b = None
        for ci in col_indices:
            cm = rg_meta.column(ci)
            s = cm.dictionary_page_offset if cm.dictionary_page_offset > 0 else cm.data_page_offset
            e = s + cm.total_compressed_size
            if rg_start_b is None or s < rg_start_b:
                rg_start_b = s
            if rg_end_b is None or e > rg_end_b:
                rg_end_b = e

        rg_len = (rg_end_b - rg_start_b) if rg_start_b is not None else 0
        return rg_start_b or 0, rg_len, all_comp

    # ── Sliding-window helpers (identical API to ParquetReaderS3Iterable) ────

    def open_footer_only(self, filename):
        """Read the parquet footer and return (pf, None, offsets) without
        submitting any prefetch futures.  Used by TorchIterableDataset's
        sliding-window iterator to separate footer reads from data reads.
        """
        if filename in self._pf_cache:
            return self._pf_cache[filename]

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filename)
        meta = pf.metadata
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)

        # rf=None: raw reads open their own fd per call (thread-safe).
        self._pf_cache[filename] = (pf, None, offsets)
        return self._pf_cache[filename]

    def submit_rg_prefetch(self, filename, rg_idx):
        """Submit a single RG raw byte-range read to the background thread pool.
        Returns the Future, or None if the executor is not available.
        Idempotent: returns the existing Future if already submitted.
        """
        key = (filename, rg_idx)
        if key in self._prefetch_futures:
            return self._prefetch_futures[key]

        pf, _, _ = self._pf_cache[filename]
        rg_start_b, rg_len, all_comp = self._rg_byte_range(pf, rg_idx)

        fut = self._prefetch_executor.submit(
            self._fetch_rg_file, filename, rg_start_b, rg_len, all_comp
        )
        self._prefetch_futures[key] = fut
        return fut

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """Return (pf, None, offsets) for filename, reading the footer at most
        once per epoch.  Eagerly submits all RG prefetches (legacy open() path).
        """
        if filename in self._pf_cache:
            return self._pf_cache[filename]

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filename)
        meta = pf.metadata
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)

        self.logger.debug(
            f"{utcnow()} ParquetReaderFileIterable.open {filename} "
            f"row_groups={meta.num_row_groups} total_rows={offsets[-1]}"
        )

        # Eagerly submit all RG reads in background threads (legacy path).
        for rg_i in range(meta.num_row_groups):
            key = (filename, rg_i)
            if key not in self._prefetch_futures:
                rg_start_b, rg_len, all_comp = self._rg_byte_range(pf, rg_i)
                self._prefetch_futures[key] = self._prefetch_executor.submit(
                    self._fetch_rg_file, filename, rg_start_b, rg_len, all_comp
                )

        self._pf_cache[filename] = (pf, None, offsets)
        return self._pf_cache[filename]

    @dlp.log
    def close(self, filename):
        """No-op: caches kept for full epoch to avoid re-reading footers.
        Flushed at epoch boundary in finalize().
        """
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """Wait on the prefetch future for the RG containing sample_index,
        record compressed byte count, discard data.
        """
        pf, _, offsets = self.open_file_map[filename]

        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        rg_idx = min(rg_idx, pf.metadata.num_row_groups - 1)

        cache_key = (filename, rg_idx)
        if cache_key not in self._rg_cache:
            if cache_key in self._prefetch_futures:
                compressed_bytes = self._prefetch_futures.pop(cache_key).result()
            else:
                # Synchronous fallback: compute byte range and read directly.
                rg_start_b, rg_len, all_comp = self._rg_byte_range(pf, rg_idx)
                self._fetch_rg_file(filename, rg_start_b, rg_len, all_comp)
                compressed_bytes = all_comp
            self._rg_cache[cache_key] = compressed_bytes

        dlp.update(image_size=self._rg_cache[cache_key])

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        filename, sample_index = self.global_index_map[image_idx]
        if (
            filename not in self.open_file_map
            or self.open_file_map[filename] is None
        ):
            self.open_file_map[filename] = self.open(filename)
        self.get_sample(filename, sample_index)
        if self._args.read_type is _ReadType.ON_DEMAND:
            self.open_file_map[filename] = None
        return self._args.resized_image

    @dlp.log
    def finalize(self):
        """Cancel outstanding futures and flush all caches at epoch boundary."""
        for fut in self._prefetch_futures.values():
            fut.cancel()
        self._prefetch_futures.clear()
        self._pf_cache.clear()
        self._rg_cache.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
