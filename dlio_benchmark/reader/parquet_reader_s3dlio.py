"""Parquet reader backed by s3dlio's PyParquetIndex.

Concurrency model — consumer-driven bounded pipeline
----------------------------------------------------
At epoch_init we build an ordered access plan (all RGs in consumption order)
and seed with exactly `window_size` in-flight GETs. read_index drives the
pipeline: it POPS the future (releasing the byte payload immediately) and
then submits the next item from the plan. Memory is strictly bounded.

  Memory guarantee:
    _active_futures holds at most window_size entries at any moment.
    Each entry holds one row-group's bytes (~8 MB) until popped.
    Peak RAM = window_size × rg_size (e.g. 32 × 8 MB = 256 MB).

  NO done_callback chaining: callbacks ran ahead of the consumer and
  accumulated ALL completed futures in memory (61 GB for 64-file DLRM).
  Consumer-driven is the only safe design.

  read_index(image_idx):
      bisect lookup  ->  (uri, rg_idx)          # pure Python, ~200 ns
      if rg_key in _rg_done: return             # fast path, ~50 ns
      fut = _active_futures.pop(rg_key)         # removes entry → frees RAM
      fut.result()                              # usually already done
      _pipeline_submit_next()                   # refill one slot

Window size:
  window_size = min(prefetch_workers, max_outstanding)
  max_outstanding = min(user_cap=1024, free_ram * 0.5 / rg_size)

Bisect table reuse:
  PyParquetIndex (Rust DashMap) retains footer metadata across epochs.
  Python bisect tables (_rg_sample_boundaries, _rg_ranges_cache) are also
  retained — the sample→RG mapping is immutable (files never change).
  Only _active_futures, _rg_done, and the plan iterator are reset per epoch.
"""
import bisect
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import StorageType
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


def _free_ram_bytes() -> int:
    """Return available physical RAM in bytes."""
    try:
        pages = os.sysconf("SC_AVAIL_PHYS_PAGES")
        page_sz = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_sz > 0:
            return pages * page_sz
    except (ValueError, OSError):
        pass
    return 8 * 1024 ** 3  # conservative fallback: 8 GiB


def _compute_max_outstanding(rg_size: int, user_cap: int = 1024) -> int:
    """
    Memory-bounded window size.
    - Small objects (< 1 MiB): cap at user_cap (default 1024).
    - Large objects: cap at 50% of free RAM / rg_size, never exceed user_cap.
    - Always at least 4.
    """
    if rg_size < 1024 * 1024:
        return user_cap
    free = _free_ram_bytes()
    budget = max(4, int(0.5 * free / rg_size))
    return min(user_cap, budget)


class ParquetReaderS3dlio(FormatReader):
    """
    Parquet reader using s3dlio.PyParquetIndex + bounded sliding-window prefetch.

    At most max_outstanding row-group GETs are in-flight at any time.
    Memory usage is strictly bounded: max_outstanding * rg_size <= 50% free RAM.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch_number):
        super().__init__(dataset_type, thread_index)

        import s3dlio
        # Partition Tokio worker threads across MPI ranks so NP processes don't
        # all claim every core.  Auto-detects OMPI_COMM_WORLD_SIZE / PMI_SIZE /
        # WORLD_SIZE; must be called before the first parquet_get_rg / get_range.
        s3dlio.configure_tokio_threads()

        opts = getattr(self._args, "storage_options", {}) or {}
        self._opts = opts
        self._epoch = epoch_number

        col_opt = opts.get("columns")
        self._columns = list(col_opt) if col_opt is not None else None

        self._num_workers = int(opts.get("prefetch_workers", opts.get("prefetch", 32)))
        self._footer_cap = int(opts.get("footer_cap", 4 * 1024 * 1024))
        self._footer_batch = int(opts.get("footer_batch_size", 16))
        # Number of consecutive row groups to coalesce into one byte-range GET.
        # E.g. coalesce_rgs=8 turns 8 × 8 MB RGs into one 64 MB request.
        # Fewer, larger requests = less HTTP overhead = higher throughput.
        self._coalesce_rgs: int = max(1, int(opts.get("coalesce_rgs", 8)))

        ep = opts.get("endpoint_url")
        if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
            os.environ["AWS_ENDPOINT_URL_S3"] = ep

        # Persistent Rust index — retained across epochs
        self._index = s3dlio.PyParquetIndex(
            col_indices=self._columns,
            footer_cap=self._footer_cap,
        )

        # URI cache: DLIO filename -> full s3:// or file:// URI
        self._uri_cache: dict[str, str] = {}

        # Per-epoch state (reset by _epoch_reset)
        self._executor: ThreadPoolExecutor | None = None
        self._active_futures: dict[tuple, Future] = {}  # (uri, group_start_rg) → Future
        self._rg_done: set = set()                       # (uri, group_start_rg) consumed this epoch
        self._max_outstanding: int = self._num_workers   # refined after footer fetch
        self._rg_size_hint: int = 8 * 1024 * 1024       # updated at first epoch_init
        self._total_bytes_read: int = 0                  # MEASURED: actual bytes from S3 this epoch
        # Plan iterator and its lock (plan advances from callback threads)
        self._plan_iter = iter([])
        self._plan_list: list = []          # full plan kept for iter_epoch
        self._plan_lock = threading.Lock()
        self._epoch_inited = False

        # Pure-Python sample→RG lookup tables built once at epoch_init.
        # _rg_sample_boundaries[uri] = sorted list of first sample index for each RG
        #   e.g. [0, 8130, 16260, ...] for 123 RGs
        # _rg_ranges[uri] = list of (offset, length) parallel to _rg_sample_boundaries
        # Lookup: bisect_right(boundaries, sample_index) - 1  →  rg_idx  (pure Python, ~200 ns)
        self._rg_sample_boundaries: dict[str, list] = {}
        self._rg_ranges_cache: dict[str, list] = {}

        # ── Simulate-IO mode ────────────────────────────────────────────────────
        # Set storage_options.simulate_io: true to skip all network I/O and
        # instead log every read_index decision to a TSV file.  Runs in
        # seconds; lets you inspect plan order, hit/fallback rates, and exact
        # access sequence without waiting for real data transfers.
        self._simulate: bool = str(opts.get("simulate_io", "false")).lower() in ("true", "1", "yes")
        self._sim_log_secs: float = float(opts.get("sim_log_secs", 60))
        self._sim_log_fh = None   # opened at _epoch_init, closed at _epoch_reset
        self._sim_log_cctx = None # zstandard compressor context (or None)
        self._sim_log_deadline: float = 0.0  # monotonic time after which we stop writing
        self._sim_plan_list: list = []  # full plan kept for simulator cursor
        self._sim_plan_pos: int = 0     # next plan slot to seed into _active_futures

        self.logger.info(
            f"{utcnow()} ParquetReaderS3dlio init thread={thread_index} "
            f"epoch={epoch_number} workers={self._num_workers} "
            f"footer_cap={self._footer_cap}"
        )

    # ── URI helpers ──────────────────────────────────────────────────────────

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

    # ── Pipeline ───────────────────────────────────────────────────────

    def _pipeline_submit_next(self) -> None:
        """
        Pull the next (key, uri, offset, length) from the plan and submit it.

        Called from _epoch_init (W times to seed) and from read_index each time
        it consumes a future. This consumer-driven approach guarantees that
        _active_futures never holds more than window_size entries simultaneously
        — so memory is bounded at window_size × rg_size regardless of how many
        total RGs the plan contains.

        No done_callback: completed futures sit in _active_futures only until
        read_index pops them. Memory is freed immediately after .result().
        """
        if self._simulate:
            # No executor in simulate mode — just advance the plan cursor and
            # store a (uri, offset, length) sentinel so read_index can log what
            # WOULD have been fetched.
            with self._plan_lock:
                if self._sim_plan_pos < len(self._sim_plan_list):
                    key, uri, offset, length = self._sim_plan_list[self._sim_plan_pos]
                    self._active_futures[key] = (uri, offset, length)
                    self._sim_plan_pos += 1
            return

        import s3dlio
        with self._plan_lock:
            item = next(self._plan_iter, None)
            if item is None:
                return
            key, uri, offset, length = item
            fut = self._executor.submit(s3dlio.get_range, uri, offset, length)
            self._active_futures[key] = fut

    # ── Epoch lifecycle ──────────────────────────────────────────────────────

    def _epoch_init(self) -> None:
        """
        Lazy init on first read_index of each epoch.

        1. Fetch all footers (no-op after epoch 1 — DashMap cache hit).
        2. Build bisect tables once (sample→RG mapping is immutable).
        3. Build ordered access plan: all RGs in the order DLIO will consume them.
        4. Create ThreadPoolExecutor and seed pipeline with window_size initial GETs.

        After this, read_index is just: bisect + set check + pop + fut.result().
        read_index drives the pipeline: each consumed RG submits the next one.
        Memory stays bounded at window_size × rg_size.
        """
        import s3dlio

        all_uris = self._all_uris()
        if not all_uris:
            self.logger.warning(f"{utcnow()} ParquetReaderS3dlio: file_list is empty")
            self._epoch_inited = True
            return

        # Footer indexing (no network I/O on epoch 2+)
        self._index.ensure_indexed(
            all_uris, epoch=self._epoch, batch_size=self._footer_batch
        )

        # Build bisect tables on first epoch only (mapping is immutable).
        if not self._rg_sample_boundaries:
            samples_per_file = getattr(self._args, "num_samples_per_file", 1_000_000)
            rg_size_hint = 8 * 1024 * 1024
            total_rgs = 0

            for uri in all_uris:
                num_rgs = self._index.file_rg_count(uri) or 0
                if num_rgs == 0:
                    self._rg_sample_boundaries[uri] = [0]
                    self._rg_ranges_cache[uri] = [(0, 0)]
                    continue

                ranges = [self._index.rg_range(uri, rg_i) for rg_i in range(num_rgs)]
                if rg_size_hint == 8 * 1024 * 1024 and ranges[0][1] > 0:
                    rg_size_hint = ranges[0][1]

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
                self._rg_ranges_cache[uri] = ranges
                total_rgs += num_rgs

            self._rg_size_hint = rg_size_hint
            user_cap = int(self._opts.get("max_outstanding", 1024))
            self._max_outstanding = _compute_max_outstanding(rg_size_hint, user_cap=user_cap)
            self.logger.info(
                f"{utcnow()} ParquetReaderS3dlio: built bisect tables: "
                f"{len(all_uris)} files, {total_rgs} RGs, "
                f"rg_size~{rg_size_hint / 1024**2:.1f} MiB"
            )

        # Build access plan: coalesced groups in FILE-MAJOR order.
        #
        # DLIO's global_index_map is file-sequential:
        #   samples 0..999999 → file0, 1000000..1999999 → file1, etc.
        # So the access order is: all groups of file0, then all of file1, etc.
        # The pipeline must prefetch file0/group0..15 before moving to file1.
        #
        # FILE-MAJOR order: for each file, all coalesced groups 0..max_groups-1.
        # With window=32 and 16 groups/file, 2 files are always fully prefetched.
        #
        # Each entry covers `coalesce_rgs` consecutive RGs as ONE byte-range
        # GET (offset of first RG .. end of last RG in group). Fewer, larger
        # requests saturate bandwidth more efficiently than many small ones.
        coalesce = self._coalesce_rgs
        coalesced_size = self._rg_size_hint * coalesce
        user_cap = int(self._opts.get("max_outstanding", 1024))
        self._max_outstanding = _compute_max_outstanding(coalesced_size, user_cap=user_cap)

        plan = []
        for uri in all_uris:
            ranges = self._rg_ranges_cache[uri]
            num_rgs = len(ranges)
            rg_start = 0
            while rg_start < num_rgs:
                rg_end = min(rg_start + coalesce, num_rgs)
                offset = ranges[rg_start][0]
                end_offset = ranges[rg_end - 1][0] + ranges[rg_end - 1][1]
                group_key = (uri, rg_start)
                plan.append((group_key, uri, offset, end_offset - offset))
                rg_start += coalesce

        self._plan_list = plan
        self._plan_iter = iter(plan)

        window_size = min(self._num_workers, self._max_outstanding)

        # ── Simulate-IO mode ────────────────────────────────────────────────
        if self._simulate:
            # Write into the DLIO output dir (hydra.run.dir) when available,
            # otherwise fall back to cwd.  The output dir is set as the
            # HydraConfig outputs.run.dir, but the simplest reliable anchor is
            # the directory of dlio.log (written by the benchmark runner).
            try:
                from hydra.core.hydra_config import HydraConfig
                _sim_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                _sim_dir = os.getcwd()
            # Write plan to TSV so we can inspect prefetch order.
            plan_path = os.path.join(_sim_dir, f"sim_plan_epoch{self._epoch}.tsv")
            with open(plan_path, "w") as pf:
                pf.write("plan_idx\turi_base\tgroup_start\toffset\tlength\n")
                for i, (key, pu, poff, plen) in enumerate(plan):
                    pf.write(f"{i}\t{os.path.basename(pu)}\t{key[1]}\t{poff}\t{plen}\n")

            # Open per-epoch access log — compressed with zstd.
            log_path = os.path.join(_sim_dir, f"sim_io_epoch{self._epoch}.tsv.zst")
            import zstandard as _zstd
            self._sim_log_cctx = _zstd.ZstdCompressor(level=1)  # level=1: fast
            _raw_fh = open(log_path, "wb")
            self._sim_log_fh = self._sim_log_cctx.stream_writer(_raw_fh, closefd=True)
            header = (
                "ts_ns\tepoch\tstep\timage_idx\tfile\tsample_idx"
                "\trg_idx\tgroup_start\tdecision\toffset\tlength\n"
            )
            self._sim_log_fh.write(header.encode())

            # Set deadline: stop writing after sim_log_secs seconds.
            self._sim_log_deadline = time.monotonic() + self._sim_log_secs

            # Seed pipeline: store sentinels (no real executor).
            self._sim_plan_list = plan
            self._sim_plan_pos = 0
            for _ in range(min(window_size, len(plan))):
                self._pipeline_submit_next()

            print(
                f"[simulate_io] epoch={self._epoch} files={len(all_uris)} "
                f"plan={len(plan)} window={window_size} "
                f"coalesce={coalesce} coalesced={coalesced_size/1024**2:.1f}MiB "
                f"log_secs={self._sim_log_secs:.0f}\n"
                f"  plan  → {plan_path}\n"
                f"  log   → {log_path}",
                flush=True,
            )
            self._epoch_inited = True
            return
        # ── End simulate-IO mode ─────────────────────────────────────────────

        self._executor = ThreadPoolExecutor(max_workers=window_size)

        # Seed the pipeline: submit exactly window_size initial GETs.
        # read_index will submit the next item each time it pops one.
        for _ in range(min(window_size, len(plan))):
            self._pipeline_submit_next()

        self.logger.info(
            f"{utcnow()} ParquetReaderS3dlio: pipeline seeded — "
            f"{len(plan)} coalesced GETs ({coalesce} RGs each, "
            f"~{coalesced_size / 1024**2:.0f} MiB/GET), "
            f"{window_size} in-flight window, "
            f"~{window_size * coalesced_size / 1024**2:.0f} MiB peak in-flight"
        )
        self._epoch_inited = True

    def _epoch_reset(self) -> None:
        """Tear down per-epoch state; called from finalize()."""
        if self._sim_log_fh is not None:
            try:
                self._sim_log_fh.flush()
                self._sim_log_fh.close()  # also flushes/closes the underlying file
            except Exception:
                pass
            self._sim_log_fh = None
            self._sim_log_cctx = None
        self._sim_log_deadline = 0.0
        self._sim_plan_list = []
        self._sim_plan_pos = 0
        self._plan_list = []
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._active_futures.clear()
        self._rg_done.clear()
        self._plan_iter = iter([])  # stop any in-flight callbacks from advancing
        self._total_bytes_read = 0
        # Bisect tables are file-static — keep across epochs
        self._epoch_inited = False

    def iter_epoch(self, file_list, batch_size):
        """
        Iterable-dataset path: called once per worker per epoch with the worker's
        FILE SHARD (already split by TorchIterableDataset.__iter__).

        Installing the shard before _epoch_init() means each worker fetches
        footers and seeds the pipeline only for ~N/W files instead of all N,
        eliminating the W× burst that caused OOM in the map-style path.

        Drives the consumer-driven sliding-window pipeline directly from the
        plan list, yielding one dummy sample per complete batch.
        """
        # Install shard — _epoch_init() calls _all_uris() which reads _file_list
        self._file_list = list(file_list)
        if self._epoch_inited:
            self._epoch_reset()
        self._epoch_init()

        sample_buf = 0
        for key, uri, offset, length in self._plan_list:
            group_start = key[1]
            coalesce = self._coalesce_rgs
            ranges = self._rg_ranges_cache[uri]
            num_rgs_in_file = len(ranges)
            rg_end = min(group_start + coalesce, num_rgs_in_file)

            # Count samples covered by this coalesced group
            boundaries = self._rg_sample_boundaries[uri]
            if rg_end < num_rgs_in_file:
                group_samples = boundaries[rg_end] - boundaries[group_start]
            else:
                samples_per_file = getattr(self._args, 'num_samples_per_file', 0)
                group_samples = (samples_per_file - boundaries[group_start]) if samples_per_file else 1

            # Consume pipeline slot: pop future, wait, submit next
            rg_key = key
            if self._simulate:
                self._active_futures.pop(rg_key, None)
            else:
                fut = self._active_futures.pop(rg_key, None)
                if fut is not None:
                    fut.result()
                else:
                    # Safety fallback: pipeline missed — fetch directly
                    import s3dlio
                    s3dlio.get_range(uri, offset, length)
                self._total_bytes_read += length
            self._rg_done.add(rg_key)
            self._pipeline_submit_next()

            sample_buf += group_samples
            while sample_buf >= batch_size:
                yield self._args.resized_image
                sample_buf -= batch_size

        self._epoch_reset()
        self._epoch += 1

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
        boundaries = self._rg_sample_boundaries[uri]
        rg_idx = bisect.bisect_right(boundaries, sample_index) - 1
        _, length = self._rg_ranges_cache[uri][rg_idx]
        rg_key = (uri, rg_idx)
        if rg_key not in self._rg_done:
            self._rg_done.add(rg_key)
            self._active_futures[rg_key].result()
        dlp.update(image_size=length)

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

        boundaries = self._rg_sample_boundaries[uri]
        rg_idx = bisect.bisect_right(boundaries, sample_index) - 1
        _, length = self._rg_ranges_cache[uri][rg_idx]

        # Map rg_idx → coalesced group key: (uri, first_rg_in_group)
        coalesce = self._coalesce_rgs
        group_start = (rg_idx // coalesce) * coalesce
        rg_key = (uri, group_start)

        # ── Simulate-IO mode ────────────────────────────────────────────────
        if self._simulate:
            if rg_key in self._rg_done:
                decision = "DONE"
                log_offset, log_length = 0, 0
            elif rg_key in self._active_futures:
                sentinel = self._active_futures.pop(rg_key)
                self._rg_done.add(rg_key)
                self._pipeline_submit_next()
                decision = "HIT"
                log_offset, log_length = sentinel[1], sentinel[2]
            else:
                # FALLBACK — pipeline miss.
                ranges = self._rg_ranges_cache[uri]
                rg_end = min(group_start + coalesce, len(ranges))
                log_offset = ranges[group_start][0]
                log_length = ranges[rg_end - 1][0] + ranges[rg_end - 1][1] - log_offset
                self._rg_done.add(rg_key)
                self._pipeline_submit_next()
                decision = "FALLBACK"
            if self._sim_log_fh is not None:
                if time.monotonic() > self._sim_log_deadline:
                    try:
                        self._sim_log_fh.flush()
                        self._sim_log_fh.close()
                    except Exception:
                        pass
                    self._sim_log_fh = None
                    print(f"[simulate_io] log closed after {self._sim_log_secs:.0f}s", flush=True)
                else:
                    line = (
                        f"{time.monotonic_ns()}\t{self._epoch}\t{step}\t{image_idx}"
                        f"\t{os.path.basename(filename)}\t{sample_index}"
                        f"\t{rg_idx}\t{group_start}\t{decision}\t{log_offset}\t{log_length}\n"
                    )
                    self._sim_log_fh.write(line.encode())
            dlp.update(image_size=length)
            return self._args.resized_image
        # ── End simulate-IO mode ─────────────────────────────────────────────

        # Fast path: group already consumed this epoch — just return.
        if rg_key in self._rg_done:
            dlp.update(image_size=length)
            return self._args.resized_image

        self._rg_done.add(rg_key)
        # Measure actual bytes in this coalesced GET (always, regardless of pipeline hit/miss).
        ranges = self._rg_ranges_cache[uri]
        rg_end = min(group_start + coalesce, len(ranges))
        coalesced_bytes = ranges[rg_end - 1][0] + ranges[rg_end - 1][1] - ranges[group_start][0]
        self._total_bytes_read += coalesced_bytes
        # Pop (not get): releases the coalesced payload bytes immediately.
        # Memory stays bounded at window_size × coalesced_size.
        fut = self._active_futures.pop(rg_key, None)
        if fut is not None:
            fut.result()
        else:
            # Safety fallback: shouldn't happen — pipeline is always ahead.
            import s3dlio
            offset = ranges[group_start][0]
            end_offset = ranges[rg_end - 1][0] + ranges[rg_end - 1][1]
            s3dlio.get_range(uri, offset, end_offset - offset)
        # Advance the pipeline: consumed one slot, submit the next.
        self._pipeline_submit_next()

        dlp.update(image_size=length)
        return self._args.resized_image

    @dlp.log
    def finalize(self):
        # Report measured bytes/sample back to DLIO so statscounter computes
        # real I/O MiB/s instead of the synthetic record_length estimate.
        # Must happen before _epoch_reset() clears _total_bytes_read.
        total_samples = self._args.num_samples_per_file * self._args.num_files_train
        if total_samples > 0 and self._total_bytes_read > 0:
            self._args.record_length = self._total_bytes_read // total_samples
            self.logger.debug(
                f"{utcnow()} ParquetReaderS3dlio epoch {self._epoch}: "
                f"measured {self._total_bytes_read / 1024**3:.3f} GiB read, "
                f"{self._args.record_length} bytes/sample"
            )
        self._epoch_reset()
        self._epoch += 1
        # Retain _index and _uri_cache across epochs — footers never change
        self.open_file_map.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
