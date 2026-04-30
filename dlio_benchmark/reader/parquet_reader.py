"""
Parquet reader for local and network filesystems (non-object-storage).

Reads parquet files via pyarrow directly. Each file is opened by reading its
footer (column + row-group metadata), then individual row groups are fetched on
demand as DLIO requests specific sample indices.

Memory policy: this is a STORAGE BENCHMARK. After a row group is read from disk
the pyarrow Table is discarded immediately. Only the compressed byte count (an
int) is kept per row group for dlp telemetry. The Parquet footer (metadata) is
the only structured data held in memory. No sample data is ever retained.

This reader is the filesystem counterpart to ParquetReaderS3Iterable.

Configuration (under storage_options in the DLIO YAML):
  columns:  null  # list of column names to read (null = all)

Example YAML snippet:
  dataset:
    format: parquet
    storage_type: local
    num_samples_per_file: 1024  # must equal actual rows-per-parquet-file
    storage_options:
      columns: ["feature1", "label"]
"""
import bisect

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ParquetReader(FormatReader):
    """
    Row-group-granular Parquet reader for local/network filesystems.

    Opens parquet files with pyarrow natively (no object-storage adapters needed).
    Row groups are read for I/O measurement, then the pyarrow Table is discarded
    immediately (del). Only the compressed byte count (int) is retained for
    dlp telemetry. The Parquet footer is the only thing held in memory.

    DLIO's FormatReader protocol:
      open(filename)            → returns (ParquetFile, cumulative_offsets)
      get_sample(filename, idx) → bisect-locates the row group, reads+discards,
                                  records byte count, updates dlp metrics
      close(filename)           → removes byte-count cache for that file
      next() / read_index()     → delegate to FormatReader base class
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        opts = getattr(self._args, "storage_options", {}) or {}

        # Optional column selection (list[str] or None = all columns)
        self._columns = opts.get("columns") or None

        # Footer cache: filename → (ParquetFile, cumulative_offsets).
        # Holds ONLY the Parquet footer metadata (schema + row-group offsets),
        # a few KB per file. Keyed by filename so open() never re-reads the
        # footer from disk. Flushed at finalize() (epoch boundary).
        # In ON_DEMAND mode the base class calls open()/close() around every
        # single sample — without this cache that means one footer read per sample.
        self._pf_cache: dict = {}

        # Row-group byte-count cache: (filename, rg_idx) → int (compressed bytes).
        # Tables are read for I/O measurement then discarded; only the byte count
        # is kept for dlp telemetry. Memory per entry is negligible (~100 bytes).
        # Flushed at finalize() (epoch boundary).
        self._rg_cache: dict = {}

        self.logger.info(
            f"{utcnow()} ParquetReader thread={thread_index} epoch={epoch} "
            f"columns={self._columns}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """
        Return the (ParquetFile, cumulative_offsets) for this file, reading the
        footer at most ONCE per epoch.

        The footer is cached in _pf_cache for the lifetime of the epoch. In
        ON_DEMAND mode the base class calls open() before every single sample;
        without this cache that means one footer read per sample.
        """
        if filename in self._pf_cache:
            return self._pf_cache[filename]

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filename)
        meta = pf.metadata

        # Build cumulative row offsets [0, rg0_rows, rg0+rg1_rows, ...]
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)

        self.logger.debug(
            f"{utcnow()} ParquetReader.open {filename} "
            f"row_groups={meta.num_row_groups} total_rows={offsets[-1]}"
        )
        self._pf_cache[filename] = (pf, offsets)
        return self._pf_cache[filename]

    @dlp.log
    def close(self, filename):
        """No-op: footer and byte-count caches are kept for the full epoch.

        In ON_DEMAND mode the base class calls close() after every single sample.
        We must NOT evict either _pf_cache (footer) or _rg_cache (byte counts)
        here — doing so forces a full footer re-read and row-group re-fetch for
        every subsequent sample on the same file.
        Both caches are flushed at epoch boundary in finalize().
        """
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Read the row group containing sample_index and update I/O metrics.

        Uses bisect to locate the row group in O(log N), fetches from disk if
        not already cached. Reports compressed row-group bytes to the profiler.
        Actual row data is discarded — DLIO uses self._args.resized_image.
        """
        pf, offsets = self.open_file_map[filename]

        # Binary search: offsets[rg_idx] <= sample_index < offsets[rg_idx+1]
        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        rg_idx = min(rg_idx, pf.metadata.num_row_groups - 1)

        cache_key = (filename, rg_idx)
        if cache_key not in self._rg_cache:
            # Read row group from disk — this is the I/O being benchmarked.
            table = pf.read_row_group(rg_idx, columns=self._columns)
            rg_meta = pf.metadata.row_group(rg_idx)
            compressed_bytes = sum(
                rg_meta.column(c).total_compressed_size
                for c in range(rg_meta.num_columns)
            )
            del table  # discard immediately — we are NOT a training framework
            self._rg_cache[cache_key] = compressed_bytes  # int only; negligible RAM

        dlp.update(image_size=self._rg_cache[cache_key])

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        """Flush both caches at epoch boundary."""
        self._pf_cache.clear()
        self._rg_cache.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
