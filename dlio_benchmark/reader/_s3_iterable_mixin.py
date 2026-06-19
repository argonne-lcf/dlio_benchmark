"""
_S3IterableMixin — shared prefetch logic for S3 iterable readers.

WHY THIS EXISTS — BENCHMARK DESIGN RATIONALE
============================================
DLIO is a storage benchmark, not a training framework. FormatReader.next() always
yields ``self._args.resized_image`` — a single pre-allocated random tensor created
ONCE at startup in ConfigArguments. The actual decoded file bytes are NEVER used in
the training loop. They are only consulted for one thing: the ``image_size`` metric
inside ``dlp.update(image_size=N)`` and ``dft_ai.update(image_size=N)``.

Therefore:
  - Calling ``np.load(BytesIO(raw))`` on NPY/NPZ data is pure CPU overhead.
  - Calling ``PIL.Image.open(BytesIO(raw))`` on JPEG/PNG data is pure CPU overhead.
  - Both allocate and immediately discard arrays that nobody reads.
  - The only value we need is ``len(raw_bytes)`` for the telemetry metric.

This mixin eliminates ALL decoding. Each prefetch method stores only the raw
byte count (int) per object key. ``get_sample()`` in each subclass uses that int
directly for telemetry — no numpy, no PIL, no intermediate allocations.

I/O IS STILL FULLY MEASURED
============================
The full network transfer still happens inside the prefetch methods (one GET per
object). Timing starts at the beginning of ``next()`` / ``read_index()`` and ends
when ``FormatReader.next()`` yields a batch. The byte-count caching only eliminates
the CPU decode after the bytes arrive, which is outside the storage bottleneck.

USAGE PATTERN
=============
Subclass from BOTH the format-specific parent AND this mixin::

    class NPYReaderS3Iterable(NPYReader, _S3IterableMixin):
        @dlp.log_init
        def __init__(self, dataset_type, thread_index, epoch):
            super().__init__(dataset_type, thread_index, epoch)
            opts = getattr(self._args, "storage_options", {}) or {}
            self._s3_init(opts)
            self.logger.info(...)

        @dlp.log
        def open(self, filename):
            return self._object_cache.get(filename)

        @dlp.log
        def close(self, filename):
            self._object_cache.pop(filename, None)

        @dlp.log
        def get_sample(self, filename, sample_index):
            dlp.update(image_size=self._object_cache.get(filename, 0))

        def next(self):
            yield from self._s3_stream_next()

        @dlp.log
        def read_index(self, image_idx, step):
            filename, _ = self.global_index_map[image_idx]
            self._s3_ensure_cached(filename)
            dlp.update(step=step)
            return super().read_index(image_idx, step)

SUPPORTED LIBRARIES (strictly isolated — no cross-library fallback)
====================================================================
  s3dlio           — get_many(); len(BytesView) is O(1), no Python bytes copy.
  s3torchconnector — S3IterableDataset.from_objects() + sequential reader;
                     reader.read() consumes the I/O; len() records byte count.
  minio            — ThreadPoolExecutor + Minio.get_object(); len(resp.read()).

The configured library is validated at construction time (_s3_init). Misconfigured
or missing libraries raise ImportError immediately, not later during I/O.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from dlio_benchmark.utils.utility import utcnow

_PREFETCH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s3_prefetch")


class _S3IterableMixin:
    """
    Mixin providing parallel S3 prefetch for NPY/NPZ/JPEG/PNG iterable readers.

    Do NOT instantiate directly. Mix in alongside a FormatReader subclass; call
    ``_s3_init(opts)`` from the subclass ``__init__`` after ``super().__init__()``.
    """

    # ── Construction-time setup ───────────────────────────────────────────────

    def _s3_init(self, opts: dict) -> None:
        """
        Validate and cache S3 connection state.

        Call from subclass ``__init__`` after ``super().__init__()``. Sets:
          - ``self._storage_library`` (str)
          - ``self._opts`` (dict)
          - ``self._object_cache`` (dict: obj_key → int byte count)
          - ``self._minio_client`` (None; lazily created on first minio prefetch)

        Raises ``ImportError`` immediately if the configured library is not
        installed, rather than deferring failure to the first I/O call.
        """
        # Default to s3dlio — consistent with how data is generated.  Users can
        # override by setting storage_library in storage_options.
        self._storage_library: str = opts.get("storage_library") or "s3dlio"
        self._opts: dict = opts
        self._object_cache: dict = {}   # obj_key → int (raw byte count only)
        self._minio_client = None       # cached across epochs for TCP keep-alive
        # Actual bytes received from storage this epoch (reset in finalize_s3_bytes).
        # Incremented in _prefetch() from real len(data) — not a configured estimate.
        self._total_bytes_read: int = 0
        self._total_objects_read: int = 0

        if self._storage_library == "s3dlio":
            # s3dlio reads AWS_ENDPOINT_URL_S3 at import time; set early.
            ep = opts.get("endpoint_url")
            if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
                os.environ["AWS_ENDPOINT_URL_S3"] = ep

        elif self._storage_library == "s3torchconnector":
            try:
                from s3torchconnector import S3IterableDataset as _DS      # noqa: F401
                from s3torchconnector.s3reader import S3ReaderConstructor as _RC  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    f"{self.__class__.__name__}: storage_library='s3torchconnector' "
                    "requires the s3torchconnector package. "
                    "Install with: pip install s3torchconnector"
                ) from exc

        elif self._storage_library == "minio":
            try:
                from minio import Minio as _Minio  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    f"{self.__class__.__name__}: storage_library='minio' requires "
                    "the minio package. Install with: pip install minio"
                ) from exc

        # (unknown library values are caught at _prefetch() time with ValueError)

    # ── URI helper ────────────────────────────────────────────────────────────

    def _uri_for_obj_key(self, obj_key: str) -> str:
        """Return a full URI for a DLIO object key, using the configured uri_scheme."""
        if "://" in obj_key:
            return obj_key
        scheme = self._opts.get("uri_scheme", "s3")
        root = self._args.storage_root.rstrip("/")
        return f"{scheme}://{root}/{obj_key.lstrip('/')}"

    # ── Minio client (cached for TCP keep-alive across epochs) ───────────────

    def _get_minio_client(self):
        """
        Return a cached Minio client with a persistent urllib3 connection pool.

        Created ONCE per worker process (lazy), reused across all epochs.
        Avoids rebuilding the urllib3 PoolManager and tearing down TCP connections
        on every prefetch call. maxsize=16 matches max_workers=16 so no thread
        ever blocks waiting for a free connection slot.

        Raises ``ImportError`` if the minio package is not installed.
        """
        if self._minio_client is not None:
            return self._minio_client

        try:
            from minio import Minio
            import urllib3
        except ImportError as exc:
            raise ImportError(
                f"{self.__class__.__name__}: storage_library='minio' requires "
                "the minio package. Install with: pip install minio"
            ) from exc

        opts = self._opts
        endpoint = opts.get("endpoint_url", "")
        if endpoint.startswith("https://"):
            host = endpoint[8:]
            secure = True
        elif endpoint.startswith("http://"):
            host = endpoint[7:]
            secure = False
        else:
            host = endpoint
            secure = False

        access_key = opts.get("access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = opts.get("secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")

        pool_kwargs = dict(
            timeout=urllib3.Timeout(connect=300, read=300),
            maxsize=16,
            retries=urllib3.Retry(
                total=5,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504],
            ),
        )
        if secure:
            import certifi
            ca_bundle = os.environ.get("AWS_CA_BUNDLE") or certifi.where()
            pool = urllib3.PoolManager(
                cert_reqs="CERT_REQUIRED", ca_certs=ca_bundle, **pool_kwargs
            )
        else:
            pool = urllib3.PoolManager(cert_reqs="CERT_NONE", **pool_kwargs)

        self._minio_client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=opts.get("region", "us-east-1"),
            http_client=pool,
        )
        return self._minio_client

    # ── Per-library prefetch — byte-count only, no numpy/PIL decode ──────────

    def _prefetch_s3dlio(self, obj_keys: list) -> dict:
        """
        Fetch all objects in parallel using ``s3dlio.get_many()``.

        s3dlio returns a ``BytesView`` (zero-copy Rust buffer). ``len()`` is O(1)
        and does not allocate a Python ``bytes`` object. No numpy decode.
        """
        import s3dlio

        uris = [self._uri_for_obj_key(k) for k in obj_keys]
        uri_to_key = dict(zip(uris, obj_keys))
        max_in_flight = min(64, len(uris))
        results = s3dlio.get_many(uris, max_in_flight=max_in_flight)

        cache = {}
        for uri, data in results:
            cache[uri_to_key.get(uri, uri)] = len(data)  # byte count only
        return cache

    def _prefetch_s3torchconnector(self, obj_keys: list) -> dict:
        """
        Fetch all objects via ``S3IterableDataset`` (one sequential GET per object).

        ``reader.read()`` consumes the full S3 transfer; ``len()`` records the byte
        count. No numpy decode. s3dlio is not referenced in any way.

        Iteration order of ``S3IterableDataset`` matches the order of ``uris``, so
        ``zip(obj_keys, dataset)`` is a safe one-to-one pairing.
        """
        from s3torchconnector import S3IterableDataset
        from s3torchconnector.s3reader import S3ReaderConstructor

        opts = self._opts
        uris = [self._uri_for_obj_key(k) for k in obj_keys]
        dataset = S3IterableDataset.from_objects(
            uris,
            region=opts.get("region", "us-east-1"),
            endpoint=opts.get("endpoint_url", ""),
            reader_constructor=S3ReaderConstructor.sequential(),
        )

        cache = {}
        for obj_key, reader in zip(obj_keys, dataset):
            cache[obj_key] = len(reader.read())   # consume I/O; discard contents
        return cache

    def _prefetch_minio(self, obj_keys: list) -> dict:
        """
        Fetch all objects concurrently via Minio SDK + ``ThreadPoolExecutor``.

        Uses a cached Minio client (TCP keep-alive across epochs).
        ``len(resp.read())`` records the byte count. No numpy decode.
        """
        client = self._get_minio_client()

        def _fetch_one(obj_key):
            uri = self._uri_for_obj_key(obj_key)
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            resp = client.get_object(bucket, key)
            try:
                raw = resp.read()
            finally:
                resp.close()
                resp.release_conn()
            return obj_key, len(raw)                  # byte count only

        n_workers = min(16, max(1, len(obj_keys)))
        cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for obj_key, byte_count in pool.map(_fetch_one, obj_keys):
                cache[obj_key] = byte_count
        return cache

    def _prefetch(self, obj_keys: list) -> dict:
        """Dispatch to the configured library's prefetch method and accumulate byte counts."""
        lib = self._storage_library
        if lib == "s3dlio":
            cache = self._prefetch_s3dlio(obj_keys)
        elif lib == "s3torchconnector":
            cache = self._prefetch_s3torchconnector(obj_keys)
        elif lib == "minio":
            cache = self._prefetch_minio(obj_keys)
        else:
            raise ValueError(
                f"{self.__class__.__name__}: unknown storage_library {lib!r}; "
                "supported: s3dlio, s3torchconnector, minio"
            )
        # Accumulate ACTUAL bytes transferred (real sizes, not configured estimates).
        self._total_bytes_read += sum(cache.values())
        self._total_objects_read += len(cache)
        return cache

    # ── FormatReader lifecycle helpers ────────────────────────────────────────

    def _s3_stream_s3dlio(self, obj_keys: list):
        """
        True sliding-window streaming via ``s3dlio.PyBytesAsyncDataLoader``.

        HOW IT WORKS
        ============
        Rather than Python managing chunk boundaries, the Rust/Tokio runtime owns
        the concurrency window entirely:

        1.  ``PyDataset.from_uris(uris)`` — builds a map-style dataset from the
            pre-computed URI list.  No network calls yet; no Python listing.
        2.  ``PyBytesAsyncDataLoader(ds, {"prefetch": N})`` — spawns a single
            Tokio producer task that keeps *exactly* N GETs in flight at all times
            via ``buffer_unordered(N)``.
        3.  ``item_iter.collect_batch(n)`` — drains up to ``n`` completed items
            from the Rust channel with **one** GIL crossing per batch rather than
            one crossing per item.  Inside the single ``py.detach()`` block, Rust
            calls ``blocking_recv()`` up to ``n`` times; on return the GIL is
            reacquired once to wrap the results into a Python list of
            ``PyObjectItem``.  Python then iterates a plain Python list —
            cheapest possible iteration, no ``__next__`` dispatch overhead.

        WHY THIS IS BETTER THAN THE CHUNKED PATH
        =========================================
        The chunked path (``_s3_stream_next``) fetches N files, then yields N
        items, then fetches the next N.  When I/O is fast (low latency or high
        bandwidth) the fetch completes in milliseconds but the yield loop takes
        hundreds of milliseconds, leaving the storage backend idle between chunks.

        This path has NO chunk boundary: Tokio always has ``prefetch`` GETs in
        flight.  The storage server sees a flat, constant request rate regardless
        of how fast or slow the compute side is.

        RUST TELLS PYTHON WHAT ARRIVED
        ================================
        ``PyObjectItem.uri`` carries the full URI of the completed GET.  Python
        never needs to track indices or maintain a parallel key list — it just
        calls ``get_sample(obj_key, s)`` for whatever arrived.  Byte count is
        available as ``len(item)`` for telemetry without a Python ``bytes`` copy.

        COLLECT BATCH SIZE
        ==================
        ``collect_n = max(1, batch_size // num_samples_per_file)`` — one training
        batch worth of files per Rust drain.  For RetinaNet (num_spf=1) this is
        exactly ``batch_size`` files.  For NPZ (num_spf=4) it is
        ``batch_size // 4`` files, still aligning each drain with one training
        batch.  Tail-latency impact of waiting for the slowest of ``collect_n``
        concurrent GETs is negligible at loopback latencies and acceptable at
        typical S3 latencies.

        Configure in-flight depth via ``storage_options.prefetch_window`` (default 64).
        """
        import s3dlio

        args = self._args
        batch_size = args.batch_size
        num_spf = args.num_samples_per_file
        opts = getattr(args, "storage_options", {}) or {}
        dummy = args.resized_image
        # prefetch_window controls Tokio in-flight GETs per worker.
        # Default 64 — matches s3dlio's max_in_flight, so no backlog builds up.
        prefetch = int(opts.get("prefetch_window", 64))
        # collect_n: files to drain per collect_batch() call (one GIL crossing).
        # Aligns to one training batch worth of files so the drain is a natural
        # unit. Falls back to 1 when batch_size < num_spf (rare).
        collect_n = max(1, batch_size // num_spf)

        uris = [self._uri_for_obj_key(k) for k in obj_keys]
        uri_to_key = {u: k for k, u in zip(obj_keys, uris)}

        total = len(uris)
        self.logger.info(
            f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
            f"s3dlio sliding-window: {total} objects, {prefetch} in-flight, "
            f"collect_batch={collect_n}"
        )

        ds = s3dlio.PyDataset.from_uris(uris)

        # skip_head controls whether s3dlio issues a HEAD before each GET.
        # Default (True): skip HEAD, do plain GET, cache the size — epoch 2+
        # will range-split correctly for large objects automatically.
        # False: issue HEAD first so range splitting can fire from epoch 1.
        # We override to False only when we know objects are >= 16 MiB, since
        # those benefit most from parallel range GETs on the very first epoch.
        record_bytes = getattr(args, "record_length", 0) or 0
        skip_head = not (record_bytes >= 16 * 1024 * 1024)
        loader_opts = {"prefetch": prefetch, "skip_head": skip_head}
        if not skip_head:
            self.logger.info(
                f"{utcnow()} {self.__class__.__name__} "
                f"skip_head=False (record_length={record_bytes} bytes >= 16 MiB, "
                f"range-split active from epoch 1)"
            )

        loader = s3dlio.PyBytesAsyncDataLoader(ds, loader_opts)
        item_iter = loader.items()

        sample_buf = 0
        # collect_batch() releases the GIL once, drains collect_n items from the
        # Rust channel (blocking_recv × collect_n without touching the GIL), then
        # reacquires the GIL once to return a Python list.  Python iterates the
        # list — zero __next__() overhead.  Empty list signals end of stream.
        while batch := item_iter.collect_batch(collect_n):
            for item in batch:
                # item.uri tells us which object arrived (completion order).
                # len(item) is byte_count — O(1), no Python bytes copy needed.
                obj_key = uri_to_key.get(item.uri, item.uri)

                # Store byte count in object cache so get_sample() / telemetry
                # can read it.
                self._object_cache[obj_key] = len(item)
                self._total_bytes_read += len(item)
                self._total_objects_read += 1

                for s in range(num_spf):
                    self.get_sample(obj_key, s)   # dlp + dft_ai image_size telemetry
                    sample_buf += 1
                    if sample_buf >= batch_size:
                        yield dummy
                        sample_buf -= batch_size

                # Release the byte-count entry; not needed across items.
                self._object_cache.pop(obj_key, None)
        # Drop-last: remaining sample_buf < batch_size is silently discarded.

    def _s3_stream_next(self):
        """
        Bounded streaming main entry point for all S3 iterable readers.

        Dispatches to the best path for the configured storage library:

        s3dlio
            Uses ``_s3_stream_s3dlio()``: the Rust/Tokio runtime keeps a sliding
            window of ``prefetch_window`` (default 64) GETs permanently in flight.
            No chunk boundaries, no Python-side threading, no idle gaps.

        minio / s3torchconnector
            Uses the chunked pipelined path: files are fetched in chunks of
            ``prefetch_window`` (default 256), with the next chunk pre-fetched in
            a background ThreadPoolExecutor thread while the current chunk is
            being yielded to the training loop.
        """
        args = self._args
        batch_size = args.batch_size
        num_spf = args.num_samples_per_file
        opts = getattr(args, "storage_options", {}) or {}
        dummy = args.resized_image

        thread_entries = self.file_map.get(self.thread_index, [])
        # Build deduplicated key list preserving epoch-shuffle order.
        seen = set()
        obj_keys = []
        for _, obj_key, _ in thread_entries:
            if obj_key not in seen:
                seen.add(obj_key)
                obj_keys.append(obj_key)

        if not obj_keys:
            return

        # s3dlio: hand control to Rust — no stagger, no chunks needed.
        if self._storage_library == "s3dlio":
            yield from self._s3_stream_s3dlio(obj_keys)
            return

        # ── Other libraries: chunked pipelined path ───────────────────────────
        # Worker stagger: spread startup I/O across one GPU-cycle window so all
        # workers don't submit their first chunk simultaneously.
        # Disable with storage_options.stagger_workers: false.
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

        window = int(opts.get("prefetch_window", 256))
        total = len(obj_keys)
        n_chunks = (total + window - 1) // window
        self.logger.info(
            f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
            f"streaming {total} objects in {n_chunks} chunks of {window} "
            f"via [{self._storage_library}]"
        )

        sample_buf = 0
        chunks = [obj_keys[i : i + window] for i in range(0, total, window)]

        # Prime the pipeline: start fetching chunk 0 in background.
        future = _PREFETCH_POOL.submit(self._prefetch, chunks[0])

        for idx, chunk in enumerate(chunks):
            cache = future.result()
            if idx + 1 < len(chunks):
                future = _PREFETCH_POOL.submit(self._prefetch, chunks[idx + 1])
            else:
                future = None

            self._object_cache = cache
            for obj_key in chunk:
                for s in range(num_spf):
                    self.get_sample(obj_key, s)
                    sample_buf += 1
                    if sample_buf >= batch_size:
                        yield dummy
                        sample_buf -= batch_size
            self._object_cache = {}
        # Drop-last: remaining sample_buf < batch_size is silently discarded.

    def _s3_prefetch_all(self) -> None:
        """
        Prefetch ALL object keys for this thread in one shot.

        Used only by ``read_index()`` (on-demand map-style access) to warm the
        cache before random access begins.  The main streaming path now uses
        ``_s3_stream_next()`` instead to avoid thundering-herd GET spikes.
        """
        thread_entries = self.file_map.get(self.thread_index, [])
        seen = set()
        obj_keys = []
        for _, obj_key, _ in thread_entries:
            if obj_key not in seen:
                seen.add(obj_key)
                obj_keys.append(obj_key)

        if obj_keys:
            self.logger.info(
                f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
                f"prefetching {len(obj_keys)} objects via [{self._storage_library}]"
            )
            self._object_cache = self._prefetch(obj_keys)

    def _s3_ensure_cached(self, filename: str) -> None:
        """Fetch a single object on demand, always re-fetching from storage.

        The cache is intentionally NOT short-circuited so that every epoch
        measures real I/O.  With persistent_workers=True (still used on the
        iterable dataset paths), reusing a cached byte count from a previous
        epoch would skip the GET entirely in epochs 2+, producing invalid AU.

        This mirrors the fix applied to _localfs_ensure_cached in PR #26 —
        that fix covered the local-filesystem map-style path but the identical
        guard (``if filename not in self._object_cache``) was not removed here.
        """
        self._object_cache.update(self._prefetch([filename]))

    def finalize_s3_bytes(self) -> None:
        """
        Update ``args.record_length`` from the actual bytes transferred this epoch.

        Must be called from each reader's ``finalize()`` BEFORE resetting epoch
        state.  Mirrors the same pattern used by ``ParquetReaderS3dlio.finalize()``.

        Uses measured bytes per object (average across all objects fetched this
        epoch) rather than the configured ``record_length_bytes`` estimate.  For
        workloads with high file-size variance (e.g. UNet3D stdev ≈ 65 MiB),
        this gives a more accurate per-epoch I/O throughput figure.

        After updating, resets the epoch counters so the next epoch starts clean.
        """
        if self._total_objects_read > 0 and self._total_bytes_read > 0:
            measured = self._total_bytes_read // self._total_objects_read
            self._args.record_length = measured
            self.logger.debug(
                f"{utcnow()} {self.__class__.__name__} epoch done: "
                f"actual {self._total_bytes_read / 1024**3:.3f} GiB read, "
                f"{self._total_objects_read} objects, "
                f"{measured:,} bytes/object"
            )
        self._total_bytes_read = 0
        self._total_objects_read = 0
