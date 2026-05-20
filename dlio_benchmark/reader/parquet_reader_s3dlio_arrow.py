"""Parquet reader using s3dlio.parquet_get_rg for Rust Arrow IPC decode.

Access pattern
--------------
Identical to ParquetReaderS3dlio (index-based, sliding-window prefetch) except
that each row-group fetch calls s3dlio.parquet_get_rg(uri, rg_idx, decode="arrow")
instead of s3dlio.get_range(uri, offset, length).

The fetch + Parquet→Arrow decode happen inside the Rust async task on the Tokio
thread pool — completely off the Python GIL.  Python receives Arrow IPC bytes
(``BytesView``) which it can pass directly to PyArrow with zero additional copy.

Concurrency model — same as ParquetReaderS3dlio
------------------------------------------------
  _active_futures holds at most window_size entries (per-RG, NOT coalesced).
  Each entry holds one decoded Arrow IPC buffer until popped.
  Peak RAM = window_size × decoded_rg_size (decoded > raw; typically 1.5-2×).

  Consumer-driven pipeline: read_index pops a future (frees the buffer) and
  submits the next one.  Memory is strictly bounded.

decode_output config key
------------------------
Controls what Python does with the decoded IPC bytes:

  "none" (default for benchmarking) — discard bytes, record byte_count only.
  "pyarrow"                         — return pa.RecordBatch via pa.ipc.open_stream.
  "torch"                           — return dict of torch tensors (requires torch).

  For pure storage benchmarking use "none".
  For real training use "pyarrow" or "torch".

Selection
---------
CLI flags (passed to mlpstorage training run --params):

    storage.storage_options.storage_library=s3dlio
    storage.storage_options.decode=arrow

Both flags are required. Omitting ``decode=arrow`` routes to
ParquetReaderS3dlio (raw bytes).
"""
import bisect
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import StorageType
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


def _free_ram_bytes() -> int:
    try:
        pages = os.sysconf("SC_AVAIL_PHYS_PAGES")
        page_sz = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_sz > 0:
            return pages * page_sz
    except (ValueError, OSError):
        pass
    return 8 * 1024 ** 3


def _compute_max_outstanding(rg_size: int, user_cap: int = 512) -> int:
    if rg_size < 1024 * 1024:
        return user_cap
    free = _free_ram_bytes()
    # Arrow IPC decoded buffers are ~1.5-2× raw size; be more conservative.
    budget = max(4, int(0.35 * free / rg_size))
    return min(user_cap, budget)


class ParquetReaderS3dlioArrow(FormatReader):
    """
    Parquet reader using s3dlio.parquet_get_rg with Rust Arrow IPC decode.

    Uses the same PyParquetIndex + sliding-window prefetch architecture as
    ParquetReaderS3dlio, but each row-group GET is decoded to Arrow IPC format
    inside the Rust async task on the Tokio thread pool — completely off the
    Python GIL.

    configure decode_output to control what Python does with the result:
      "none"    (default) — discard, record byte_count only (pure benchmark)
      "pyarrow"           — return pa.RecordBatch (real training)
      "torch"             — return dict of torch tensors (real training)
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch_number):
        super().__init__(dataset_type, thread_index)

        import s3dlio
        # Partition Tokio worker threads across MPI ranks so NP processes don't
        # all claim every core.  Auto-detects OMPI_COMM_WORLD_SIZE / PMI_SIZE /
        # WORLD_SIZE; must be called before the first parquet_get_rg call.
        s3dlio.configure_tokio_threads()

        opts = getattr(self._args, "storage_options", {}) or {}
        self._opts = opts
        self._epoch = epoch_number

        col_opt = opts.get("columns")
        self._columns = list(col_opt) if col_opt is not None else None
        self._footer_cap = int(opts.get("footer_cap", 4 * 1024 * 1024))
        self._footer_batch = int(opts.get("footer_batch_size", 16))
        self._num_workers = int(opts.get("prefetch_workers", opts.get("prefetch", 32)))

        # "none"    → discard decoded bytes (pure I/O+decode benchmark)
        # "pyarrow" → return pa.RecordBatch
        # "torch"   → return dict of torch tensors
        self._decode_output: str = opts.get("decode_output", "none").lower()

        ep = opts.get("endpoint_url")
        if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
            os.environ["AWS_ENDPOINT_URL_S3"] = ep

        # Persistent Rust index — retained across epochs
        self._index = s3dlio.PyParquetIndex(
            col_indices=self._columns,
            footer_cap=self._footer_cap,
        )

        # URI cache: DLIO filename -> full URI
        self._uri_cache: dict[str, str] = {}

        # Per-epoch state (reset by _epoch_reset)
        self._executor: ThreadPoolExecutor | None = None
        self._active_futures: dict[tuple, Future] = {}   # (uri, rg_idx) → Future
        self._rg_done: set = set()
        self._max_outstanding: int = self._num_workers
        self._rg_size_hint: int = 8 * 1024 * 1024
        self._total_bytes_read: int = 0
        self._plan_iter = iter([])
        self._plan_lock = threading.Lock()
        self._epoch_inited = False

        # Bisect tables — built once per lifetime (immutable after epoch 1)
        self._rg_sample_boundaries: dict[str, list] = {}
        self._rg_counts: dict[str, int] = {}

        self.logger.info(
            f"{utcnow()} ParquetReaderS3dlioArrow init thread={thread_index} "
            f"epoch={epoch_number} workers={self._num_workers} "
            f"decode_output={self._decode_output}"
        )

    # ── URI helpers ───────────────────────────────────────────────────────────

    def _uri_for_filename(self, filename: str) -> str:
        cached = self._uri_cache.get(filename)
        if cached is not None:
            return cached
        if "://" in filename:
            uri = filename
        else:
            storage_type = getattr(self._args, "storage_type", StorageType.LOCAL_FS)
            if storage_type in (StorageType.S3, StorageType.AISTORE):
                bucket = self._args.storage_root.rstrip("/")
                uri = f"s3://{bucket}/{filename.lstrip('/')}"
            else:
                uri = f"file://{os.path.abspath(filename)}"
        self._uri_cache[filename] = uri
        return uri

    def _all_uris(self) -> list:
        return [self._uri_for_filename(f) for f in (self._file_list or [])]

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _pipeline_submit_next(self) -> None:
        import s3dlio
        with self._plan_lock:
            item = next(self._plan_iter, None)
            if item is None:
                return
            uri, rg_idx = item
            col = self._columns
            footer_cap = self._footer_cap
            fut = self._executor.submit(
                s3dlio.parquet_get_rg, uri, rg_idx, col, footer_cap, "arrow"
            )
            self._active_futures[(uri, rg_idx)] = fut

    # ── Epoch lifecycle ───────────────────────────────────────────────────────

    def _epoch_init(self) -> None:
        import s3dlio

        all_uris = self._all_uris()
        if not all_uris:
            self.logger.warning(f"{utcnow()} ParquetReaderS3dlioArrow: file_list is empty")
            self._epoch_inited = True
            return

        # Footer indexing (no network I/O on epoch 2+)
        self._index.ensure_indexed(
            all_uris, epoch=self._epoch, batch_size=self._footer_batch
        )

        # Build bisect tables once (immutable across epochs)
        if not self._rg_sample_boundaries:
            samples_per_file = getattr(self._args, "num_samples_per_file", 1_000_000)
            total_rgs = 0

            for uri in all_uris:
                num_rgs = self._index.file_rg_count(uri) or 0
                self._rg_counts[uri] = num_rgs
                if num_rgs == 0:
                    self._rg_sample_boundaries[uri] = [0]
                    continue

                rows_per_rg_est = samples_per_file // num_rgs
                probe_rg, _, _ = self._index.rg_lookup(uri, rows_per_rg_est)
                if probe_rg == 1:
                    boundaries = [rg_i * rows_per_rg_est for rg_i in range(num_rgs)]
                else:
                    boundaries = [0] * num_rgs
                    for rg_i in range(1, num_rgs):
                        lo = (rg_i - 1) * rows_per_rg_est
                        hi = min(samples_per_file - 1, (rg_i + 1) * rows_per_rg_est)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            got, _, _ = self._index.rg_lookup(uri, mid)
                            if got < rg_i:
                                lo = mid + 1
                            else:
                                hi = mid
                        boundaries[rg_i] = lo
                self._rg_sample_boundaries[uri] = boundaries
                total_rgs += num_rgs

            # Estimate decoded RG size (~1.6× raw) for memory bound calculation
            if all_uris and self._rg_counts.get(all_uris[0], 0) > 0:
                _, raw_len = self._index.rg_range(all_uris[0], 0)
                self._rg_size_hint = int(raw_len * 1.6)

            user_cap = int(self._opts.get("max_outstanding", 512))
            self._max_outstanding = _compute_max_outstanding(
                self._rg_size_hint, user_cap=user_cap
            )
            self.logger.info(
                f"{utcnow()} ParquetReaderS3dlioArrow: built bisect tables: "
                f"{len(all_uris)} files, {total_rgs} RGs, "
                f"decoded_rg_size~{self._rg_size_hint / 1024**2:.1f} MiB, "
                f"window={min(self._num_workers, self._max_outstanding)}"
            )

        # Access plan: per-RG, file-major order (no coalescing — each RG decoded
        # independently in Rust, so coalescing does not help here)
        plan = []
        for uri in all_uris:
            num_rgs = self._rg_counts.get(uri, 0)
            for rg_idx in range(num_rgs):
                plan.append((uri, rg_idx))

        self._plan_iter = iter(plan)
        window_size = min(self._num_workers, self._max_outstanding)

        self._executor = ThreadPoolExecutor(max_workers=window_size)
        for _ in range(min(window_size, len(plan))):
            self._pipeline_submit_next()

        self.logger.info(
            f"{utcnow()} ParquetReaderS3dlioArrow: pipeline seeded — "
            f"{len(plan)} RG fetches, {window_size} in-flight"
        )
        self._epoch_inited = True

    def _epoch_reset(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._active_futures.clear()
        self._rg_done.clear()
        self._plan_iter = iter([])
        self._total_bytes_read = 0
        self._epoch_inited = False

    # ── Decode helper ──────────────────────────────────────────────────────────

    def _decode_ipc(self, bv) -> object:
        """Convert Arrow IPC BytesView to the configured output format."""
        if self._decode_output == "none":
            return self._args.resized_image
        raw = bytes(bv)
        if self._decode_output == "pyarrow":
            import pyarrow as pa
            return pa.ipc.open_stream(pa.py_buffer(raw)).read_next_batch()
        if self._decode_output == "torch":
            import pyarrow as pa
            import torch
            batch = pa.ipc.open_stream(pa.py_buffer(raw)).read_next_batch()
            return {
                col: torch.from_numpy(batch.column(col).to_pydict()["values"])
                for col in batch.schema.names
            }
        return self._args.resized_image

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        return filename

    @dlp.log
    def close(self, filename):
        pass

    @dlp.log
    def get_sample(self, filename, sample_index):
        if not self._epoch_inited:
            self._epoch_init()
        uri = self._uri_for_filename(filename)
        boundaries = self._rg_sample_boundaries.get(uri, [0])
        rg_idx = bisect.bisect_right(boundaries, sample_index) - 1
        rg_key = (uri, rg_idx)

        if rg_key not in self._rg_done:
            self._rg_done.add(rg_key)
            fut = self._active_futures.pop(rg_key, None)
            if fut is not None:
                bv = fut.result()
            else:
                import s3dlio
                bv = s3dlio.parquet_get_rg(uri, rg_idx, self._columns, self._footer_cap, "arrow")
            self._total_bytes_read += len(bv)
            self._pipeline_submit_next()
            return self._decode_ipc(bv)
        return self._args.resized_image

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        if not self._epoch_inited:
            self._epoch_init()

        dlp.update(step=step)
        filename, sample_index = self.global_index_map[image_idx]
        uri = self._uri_for_filename(filename)

        boundaries = self._rg_sample_boundaries.get(uri, [0])
        rg_idx = bisect.bisect_right(boundaries, sample_index) - 1
        rg_key = (uri, rg_idx)

        if rg_key in self._rg_done:
            dlp.update(image_size=self._rg_size_hint)
            return self._args.resized_image

        self._rg_done.add(rg_key)
        fut = self._active_futures.pop(rg_key, None)
        if fut is not None:
            bv = fut.result()
        else:
            # Safety fallback: pipeline miss (should not happen in sequential access)
            import s3dlio
            bv = s3dlio.parquet_get_rg(uri, rg_idx, self._columns, self._footer_cap, "arrow")

        self._total_bytes_read += len(bv)
        dlp.update(image_size=len(bv))
        self._pipeline_submit_next()
        return self._decode_ipc(bv)

    @dlp.log
    def finalize(self):
        total_samples = self._args.num_samples_per_file * self._args.num_files_train
        if total_samples > 0 and self._total_bytes_read > 0:
            self._args.record_length = self._total_bytes_read // total_samples
            self.logger.debug(
                f"{utcnow()} ParquetReaderS3dlioArrow epoch {self._epoch}: "
                f"measured {self._total_bytes_read / 1024**3:.3f} GiB read+decoded, "
                f"{self._args.record_length} bytes/sample"
            )
        self._epoch_reset()
        self._epoch += 1
        self.open_file_map.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

