"""
_LocalFSIterableMixin — parallel prefetch for local-filesystem iterable readers.

WHY THIS EXISTS — PARITY WITH _S3IterableMixin
===============================================
DLIO is a storage benchmark. FormatReader.next() always yields
``self._args.resized_image`` — a single pre-allocated dummy tensor. The actual
decoded file bytes are NEVER used. They are consulted for exactly one thing:
the ``image_size`` metric inside ``dlp.update(image_size=N)``.

Without this mixin, local-FS readers open and decode files ONE AT A TIME inside
the next() loop (queue depth = 1). The S3 iterable readers pre-fetch ALL files
in parallel before the iteration starts (queue depth = N). This is a structural
parity violation — local-FS benchmarks look slower than they physically should
be, making cross-backend comparisons invalid.

This mixin gives local-FS readers the same pre-fetch pattern as _S3IterableMixin:

1. Before next(): parallel-read all assigned files via ThreadPoolExecutor (buffered)
   OR via s3dlio.get_many() with direct:// URIs (O_DIRECT, page-cache bypass).
2. Store only the raw byte count per file (never decode numpy/PIL/h5py)
3. During next() / get_sample(): dict lookup → telemetry → return resized_image

I/O IS FULLY MEASURED
=====================
The full read() of each file still happens inside _localfs_prefetch_all().
Only the decode step (np.load, PIL.open, h5py.File) is skipped — that decode
is pure CPU overhead that has nothing to do with storage bandwidth.

TWO PREFETCH MODES
==================
storage_library: <unset or "posix">
    ThreadPoolExecutor(64) + Python open() + buffered read.
    Simple, portable, uses OS page cache.

storage_library: "direct"
    s3dlio.get_many() with direct:// URIs.
    Uses O_DIRECT (Linux) — bypasses page cache entirely, 4 KiB-aligned I/O
    via Tokio async tasks in the s3dlio Rust runtime. GIL is released for the
    full duration of all reads.
    **Required for accurate NVMe benchmarking** — repeated buffered reads hit
    the page cache rather than the device, understating storage latency and
    saturating DRAM bandwidth instead of device bandwidth.

USAGE PATTERN
=============
Subclass from BOTH the format-specific parent AND this mixin::

    class ImageReader(_OriginalImageReader, _LocalFSIterableMixin):
        @dlp.log_init
        def __init__(self, dataset_type, thread_index, epoch):
            super().__init__(dataset_type, thread_index, epoch)
            self._localfs_init()

        @dlp.log
        def open(self, filename):
            return self._local_cache.get(filename, 0)

        @dlp.log
        def get_sample(self, filename, sample_index):
            dlp.update(image_size=self._local_cache.get(filename, 0))

        def next(self):
            self._localfs_prefetch_all()
            for batch in super().next():
                yield batch
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dlio_benchmark.utils.utility import utcnow

_PREFETCH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="localfs_prefetch")


class _LocalFSIterableMixin:
    """
    Mixin providing parallel local-filesystem prefetch for iterable readers.

    Do NOT instantiate directly. Mix in alongside a FormatReader subclass;
    call ``_localfs_init()`` from the subclass ``__init__`` after
    ``super().__init__()``.

    Set ``storage_library: direct`` in storage_options to use s3dlio's O_DIRECT
    path (bypasses page cache — essential for accurate NVMe benchmarking).
    Default (no storage_library, or ``posix``) uses buffered Python open().
    """

    def _localfs_init(self) -> None:
        """
        Initialise mixin state.

        Reads ``storage_options.storage_library`` from ConfigArguments:
          - ``"direct"`` → s3dlio O_DIRECT path (``direct://`` URIs, Tokio, GIL-free)
          - anything else → buffered Python ThreadPoolExecutor path

        Sets:
          - ``self._local_cache``      (dict: filename → int byte count)
          - ``self._use_direct``       (bool)
          - ``self._storage_root``     (str absolute path, for direct:// URI construction)
          - ``self._total_bytes_read`` (int, epoch accumulator)
          - ``self._total_objects_read`` (int, epoch accumulator)
        """
        self._local_cache: dict = {}
        self._total_bytes_read: int = 0
        self._total_objects_read: int = 0

        opts = getattr(self._args, "storage_options", {}) or {}
        lib = opts.get("storage_library", "")
        self._use_direct: bool = (lib == "direct")

        if self._use_direct:
            try:
                import s3dlio as _s3dlio  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    f"{self.__class__.__name__}: storage_library='direct' requires "
                    "the s3dlio package. Install with: pip install s3dlio"
                ) from exc

    # ── URI helpers ───────────────────────────────────────────────────────────

    def _direct_uri_for_path(self, path: str) -> str:
        """Return a ``direct://`` URI for an absolute or relative local path."""
        return f"direct://{os.path.abspath(path)}"

    # ── Buffered path (default) ───────────────────────────────────────────────

    def _read_local_bytes(self, path: str) -> int:
        """Read a local file using buffered I/O and return its byte count. No decode."""
        with open(path, 'rb') as fh:
            return len(fh.read())

    def _prefetch_buffered(self, paths: list) -> dict:
        """
        Parallel buffered reads via ThreadPoolExecutor(64).

        Uses the OS page cache. Fast for warm-cache runs; not representative of
        cold-device bandwidth on NVMe.
        """
        n_workers = min(64, len(paths))
        cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for path, byte_count in zip(paths, pool.map(self._read_local_bytes, paths)):
                cache[path] = byte_count
        return cache

    # ── O_DIRECT path (storage_library: direct) ───────────────────────────────

    def _prefetch_direct(self, paths: list) -> dict:
        """
        Parallel O_DIRECT reads via ``s3dlio.get_many()`` with ``direct://`` URIs.

        - Bypasses the OS page cache (Linux O_DIRECT, 4 KiB-aligned buffers).
        - Runs in Tokio async tasks inside the s3dlio Rust runtime; GIL is
          released for the full duration.
        - ``len(data)`` is O(1) on the returned BytesView — no Python bytes copy.
        - Up to 64 concurrent reads in flight (same as _prefetch_buffered workers).

        This is the correct mode for NVMe benchmarks: it stresses the device
        itself rather than DRAM bandwidth or page-cache eviction policy.
        """
        import s3dlio

        uris = [self._direct_uri_for_path(p) for p in paths]
        uri_to_path = dict(zip(uris, paths))
        max_in_flight = min(64, len(uris))
        results = s3dlio.get_many(uris, max_in_flight=max_in_flight)

        cache = {}
        for uri, data in results:
            path = uri_to_path.get(uri, uri)
            cache[path] = len(data)   # byte count only; BytesView.len() is O(1)
        return cache

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _localfs_stream_next(self):
        """
        Bounded streaming replacement for the ``_localfs_prefetch_all() + super().next()``
        bulk-prefetch pattern.

        PROBLEM WITH BULK-PREFETCH (identical to _S3IterableMixin._s3_prefetch_all)
        ===========================================================================
        ``_localfs_prefetch_all()`` submitted ALL files for this worker to either
        ThreadPoolExecutor (buffered) or s3dlio.get_many() (O_DIRECT) before the
        training loop started.  With many files this causes:

          Buffered   — OS schedules hundreds of concurrent read() syscalls at once;
                       page-cache lock contention on Linux degrades throughput.
          O_DIRECT   — s3dlio Tokio runtime receives all io_uring sqe submissions
                       simultaneously; the kernel io_uring queue depth (typically
                       32–4096) becomes the bottleneck rather than device bandwidth.

        SOLUTION: CHUNKED STREAMING WINDOW
        ===================================
        Files are processed in chunks of ``prefetch_window`` (default 256).
        Each chunk is read — at most 64 concurrent in-flight reads — then iterated
        for telemetry and batch assembly, then freed before the next chunk.

        This keeps the kernel I/O queue at a steady, bounded depth throughout the
        epoch rather than spiking at the start.  The behaviour mirrors
        ``_S3IterableMixin._s3_stream_next()`` exactly; workload comparisons
        between S3 and local storage remain apples-to-apples.

        Configure window size via ``storage_options.prefetch_window`` (default 256).
        Setting it to 64 matches max_in_flight exactly for tightest queue control.
        """
        args = self._args
        batch_size = args.batch_size
        num_spf = args.num_samples_per_file
        opts = getattr(args, "storage_options", {}) or {}
        window = int(opts.get("prefetch_window", 256))
        dummy = args.resized_image
        mode = "s3dlio-direct://" if self._use_direct else "buffered"

        thread_entries = self.file_map.get(self.thread_index, [])
        seen = set()
        paths = []
        for _, filename, _ in thread_entries:
            if filename not in seen:
                seen.add(filename)
                paths.append(filename)

        if not paths:
            return

        # ── Worker stagger ────────────────────────────────────────────────────
        # Same rationale as _S3IterableMixin._s3_stream_next(): without a
        # startup delay all workers submit their first chunk simultaneously.
        # For buffered reads this spikes page-cache lock contention; for
        # O_DIRECT it floods the io_uring submission queue all at once.
        # Disable by setting storage_options.stagger_workers: false.
        if str(opts.get("stagger_workers", "true")).lower() not in ("false", "0", "no"):
            ct_raw = getattr(args, "computation_time", 0.0)
            ct = ct_raw.get("mean", 0.0) if isinstance(ct_raw, dict) else float(ct_raw or 0.0)
            delay = self.thread_index * ct
            if delay > 0:
                self.logger.debug(
                    f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
                    f"stagger delay={delay:.4f}s ({self.thread_index} × {ct:.4f}s)"
                )
                time.sleep(delay)

        total = len(paths)
        n_chunks = (total + window - 1) // window
        self.logger.info(
            f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
            f"streaming {total} files in {n_chunks} chunks of {window} [{mode}]"
        )

        sample_buf = 0

        # ── Pipelined chunk loop ─────────────────────────────────────────────
        # Same rationale as _S3IterableMixin._s3_stream_next().
        # The background thread fetches chunk N+1 while the main thread yields
        # batches from chunk N.  For buffered reads the background thread holds
        # the GIL only for the open()+read() syscall setup; the kernel I/O
        # itself is blocking but runs concurrently with Python's yield loop.
        # For O_DIRECT, s3dlio releases the GIL entirely during Rust io_uring.
        def _fetch(chunk):
            if self._use_direct:
                c = self._prefetch_direct(chunk)
            else:
                c = self._prefetch_buffered(chunk)
            self._total_bytes_read += sum(c.values())
            self._total_objects_read += len(c)
            return c

        chunks = [paths[i : i + window] for i in range(0, total, window)]

        future = _PREFETCH_POOL.submit(_fetch, chunks[0])

        for idx, chunk in enumerate(chunks):
            cache = future.result()

            if idx + 1 < len(chunks):
                future = _PREFETCH_POOL.submit(_fetch, chunks[idx + 1])
            else:
                future = None

            self._local_cache = cache
            for path in chunk:
                for s in range(num_spf):
                    self.get_sample(path, s)   # dlp + dft_ai image_size update
                    sample_buf += 1
                    if sample_buf >= batch_size:
                        yield dummy
                        sample_buf -= batch_size
            self._local_cache = {}
        # Drop-last: remaining sample_buf < batch_size is silently discarded.

    def _localfs_prefetch_all(self) -> None:
        """
        Prefetch ALL files for this thread in one shot.

        Retained for ``read_index()`` (on-demand map-style access) to warm the
        cache before random access begins.  The main streaming path uses
        ``_localfs_stream_next()`` instead to avoid thundering-herd I/O.
        """
        thread_entries = self.file_map.get(self.thread_index, [])
        seen = set()
        paths = []
        for _, filename, _ in thread_entries:
            if filename not in seen:
                seen.add(filename)
                paths.append(filename)

        if not paths:
            return

        mode = "s3dlio-direct://" if self._use_direct else "buffered"
        self.logger.info(
            f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
            f"prefetching {len(paths)} local files [{mode}]"
        )

        if self._use_direct:
            cache = self._prefetch_direct(paths)
        else:
            cache = self._prefetch_buffered(paths)

        self._total_bytes_read += sum(cache.values())
        self._total_objects_read += len(cache)
        self._local_cache = cache

    def _localfs_ensure_cached(self, filename: str) -> None:
        """Read a single file on demand, always re-reading from storage.

        The cache is intentionally NOT used for map-style access so that every
        epoch measures real I/O.  With persistent_workers=True, reusing cached
        byte counts would skip all reads in epochs 2+, producing invalid AU.
        """
        if self._use_direct:
            result = self._prefetch_direct([filename])
            self._local_cache.update(result)
        else:
            self._local_cache[filename] = self._read_local_bytes(filename)
        self._total_bytes_read += self._local_cache[filename]
        self._total_objects_read += 1

    def finalize_local_bytes(self) -> None:
        """
        Update ``args.record_length`` from actual bytes read this epoch.

        Mirrors ``_S3IterableMixin.finalize_s3_bytes()``. Call from subclass
        ``finalize()`` before resetting epoch state.  Resets epoch counters.
        """
        if self._total_objects_read > 0 and self._total_bytes_read > 0:
            measured = self._total_bytes_read // self._total_objects_read
            self._args.record_length = measured
            self.logger.debug(
                f"{utcnow()} {self.__class__.__name__} epoch done: "
                f"actual {self._total_bytes_read / 1024**3:.3f} GiB read, "
                f"{self._total_objects_read} files, "
                f"{measured:,} bytes/file"
            )
        self._total_bytes_read = 0
        self._total_objects_read = 0
