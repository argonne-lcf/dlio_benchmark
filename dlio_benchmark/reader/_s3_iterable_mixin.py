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
            self._s3_prefetch_all()
            for batch in super().next():
                yield batch

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
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from dlio_benchmark.utils.utility import utcnow


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
        # storage_library is REQUIRED — there is no default.  Every object
        # storage workload must explicitly declare which library to use.
        self._storage_library: str = opts.get("storage_library")
        if self._storage_library is None:
            raise ValueError(
                "storage_options['storage_library'] is required for S3 readers. "
                "Add 'storage_library: <value>' under the 'storage:' section of "
                "your workload YAML.  Supported values: minio, s3dlio, s3torchconnector."
            )
        self._opts: dict = opts
        self._object_cache: dict = {}   # obj_key → int (raw byte count only)
        self._minio_client = None       # cached across epochs for TCP keep-alive

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
        """Dispatch to the configured library's prefetch method."""
        lib = self._storage_library
        if lib == "s3dlio":
            return self._prefetch_s3dlio(obj_keys)
        elif lib == "s3torchconnector":
            return self._prefetch_s3torchconnector(obj_keys)
        elif lib == "minio":
            return self._prefetch_minio(obj_keys)
        else:
            raise ValueError(
                f"{self.__class__.__name__}: unknown storage_library {lib!r}; "
                "supported: s3dlio, s3torchconnector, minio"
            )

    # ── FormatReader lifecycle helpers ────────────────────────────────────────

    def _s3_prefetch_all(self) -> None:
        """
        Collect all object keys assigned to this thread and prefetch them.

        Call at the top of ``next()`` to bulk-load all objects before the
        training iteration starts. Deduplicates object keys while preserving order
        (an NPZ/NPY file may contain many samples mapped to the same key).
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
        """Fetch a single object on demand if it is not already in the cache."""
        if filename not in self._object_cache:
            self._object_cache.update(self._prefetch([filename]))
