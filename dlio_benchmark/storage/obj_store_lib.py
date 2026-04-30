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
import logging
from time import time
from io import BytesIO

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.storage.s3_storage import S3Storage
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from urllib.parse import urlparse
import os

from dlio_benchmark.utils.utility import Profile

# Module-level import so unittest.mock.patch can intercept S3Client in tests.
# s3torchconnector may not be installed — None is the safe sentinel.
try:
    from s3torchconnector._s3client import S3Client, S3ClientConfig
except ImportError:
    S3Client = None       # type: ignore[assignment,misc]
    S3ClientConfig = None  # type: ignore[assignment,misc]

dlp = Profile(MODULE_STORAGE)


class MinIOAdapter:
    """Adapter to make Minio client compatible with S3Client API"""
    
    def __init__(self, endpoint, access_key, secret_key, region=None, secure=True):
        from minio import Minio
        import urllib3
        import ssl
        # Parse endpoint to extract host and determine secure
        if endpoint:
            parsed = urlparse(endpoint if '://' in endpoint else f'http://{endpoint}')
            host = parsed.netloc or parsed.path
            secure = parsed.scheme == 'https' if parsed.scheme else secure
        else:
            host = "localhost:9000"

        # When TLS is in use, honour AWS_CA_BUNDLE for self-signed certificates.
        http_client = None
        if secure:
            ca_bundle = os.environ.get("AWS_CA_BUNDLE")
            if ca_bundle:
                ctx = ssl.create_default_context(cafile=ca_bundle)
                # maxsize must be set explicitly — urllib3 2.x defaults it to 1
                # per pool. Minio uses num_parallel_uploads=3 threads for
                # multipart uploads; without maxsize>=3 all but one connection
                # is discarded on return, flooding logs with
                # "Connection pool is full, discarding connection".
                http_client = urllib3.PoolManager(ssl_context=ctx, maxsize=10)

        self.client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
            http_client=http_client,
        )
        
    def get_object(self, bucket_name, object_name, start=None, end=None):
        """Adapter for get_object to match S3Client API"""
        class MinioReader:
            def __init__(self, response):
                self.response = response
                
            def read(self):
                return self.response.read()
                
            def close(self):
                self.response.close()
                self.response.release_conn()
        
        if start is not None and end is not None:
            length = end - start + 1
            response = self.client.get_object(bucket_name, object_name, offset=start, length=length)
        else:
            response = self.client.get_object(bucket_name, object_name)
        return MinioReader(response)
    
    def put_object(self, bucket_name, object_name):
        """Adapter for put_object to match S3Client API"""
        class MinioWriter:
            def __init__(self, client, bucket, obj_name):
                self.client = client
                self.bucket = bucket
                self.obj_name = obj_name
                self.buffer = BytesIO()
                
            def write(self, data):
                if isinstance(data, bytes):
                    self.buffer.write(data)
                else:
                    self.buffer.write(data.encode())
                    
            def close(self):
                self.buffer.seek(0)
                length = len(self.buffer.getvalue())
                self.client.put_object(
                    self.bucket,
                    self.obj_name,
                    self.buffer,
                    length
                )
                self.buffer.close()
        
        return MinioWriter(self.client, bucket_name, object_name)
    
    def list_objects(self, bucket_name, prefix=None):
        """Adapter for list_objects to match S3Client API"""
        class MinioListResult:
            def __init__(self, objects, prefix):
                self.object_info = []
                for obj in objects:
                    obj_info = type('ObjectInfo', (), {'key': obj.object_name})()
                    self.object_info.append(obj_info)
                self.prefix = prefix
        
        objects = self.client.list_objects(bucket_name, prefix=prefix or "", recursive=True)
        # Convert generator to list for iteration
        obj_list = list(objects)
        return [MinioListResult(obj_list, prefix)]


class ObjStoreLibStorage(S3Storage):
    """
    Storage backend for object storage with multi-library support.

    Decoupled from any specific URI scheme: the uri_scheme is read from
    storage_options (defaulting to "s3") and applied to all URI construction
    so the same code works with s3://, az://, gs://, file://, etc.

    Supports 3 storage libraries via YAML config:
      storage_library: s3dlio           # zero-copy multi-protocol (s3/az/gs/file)
      storage_library: s3torchconnector # AWS official S3 connector
      storage_library: minio            # MinIO native SDK
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)

        # Access config values from self._args (inherited from DataStorage)
        storage_options = getattr(self._args, "storage_options", {}) or {}

        logging.debug(f"ObjStoreLibStorage.__init__: namespace={namespace!r}")
        logging.debug(f"  framework={framework!r}")
        logging.debug(f"  storage_options={storage_options!r}")
        logging.debug(f"  args.storage_type={getattr(self._args, 'storage_type', '<missing>')!r}")
        logging.debug(f"  args.storage_root={getattr(self._args, 'storage_root', '<missing>')!r}")
        logging.debug(f"  args.data_folder={getattr(self._args, 'data_folder', '<missing>')!r}")
        logging.debug(f"  args.s3_region={getattr(self._args, 's3_region', '<missing>')!r}")
        logging.debug(f"  env AWS_ENDPOINT_URL={os.environ.get('AWS_ENDPOINT_URL', '<not set>')!r}")
        logging.debug(f"  env AWS_ENDPOINT_URL_S3={os.environ.get('AWS_ENDPOINT_URL_S3', '<not set>')!r}")
        logging.debug(f"  env AWS_ACCESS_KEY_ID={'<set>' if os.environ.get('AWS_ACCESS_KEY_ID') else '<not set>'}")

        # Get storage library selection.
        # storage_library is REQUIRED — there is no default.  This value flows
        # from config.py via storage_options; it must be set explicitly in every
        # object storage workload YAML (storage_library: <value>) or on the CLI
        # (storage.storage_options.storage_library=<value>).
        storage_library = storage_options.get("storage_library")
        if storage_library is None:
            raise ValueError(
                "storage_options['storage_library'] is required for ObjStoreLibStorage. "
                "Add 'storage_library: <value>' under the 'storage:' section of your "
                "workload YAML.  Supported values: minio, s3dlio, s3torchconnector."
            )
        self.storage_library = storage_library
        
        logging.debug(f"ObjStoreLibStorage: using storage library: {storage_library}")
        
        # Get credentials and endpoint config.
        # Credentials MUST NOT be hardcoded in YAML — they come from env vars
        # (set via .env file before launching dlio_benchmark).  storage_options
        # may only contain non-sensitive settings (endpoint_url, region, etc.).
        # If the key IS present in storage_options it takes priority, which
        # allows per-run overrides without touching the YAML on disk.
        self.access_key_id = storage_options.get("access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID")
        self.secret_access_key = storage_options.get("secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.endpoint = storage_options.get("endpoint_url") or os.environ.get("AWS_ENDPOINT_URL")
        self.region = storage_options.get("region") or os.environ.get("AWS_REGION") or getattr(self._args, "s3_region", "us-east-1")

        # Multi-endpoint: if S3_ENDPOINT_URIS is set, select an endpoint based on MPI rank.
        # Each MPI rank uses a different endpoint (round-robin assignment), distributing
        # write load across all servers. Works for s3dlio, s3torchconnector, and minio.
        # Example: S3_ENDPOINT_URIS='http://10.0.0.1:9000,http://10.0.0.2:9000'
        _ep_uris_str = os.environ.get("S3_ENDPOINT_URIS", "").strip()
        if _ep_uris_str:
            _ep_list = [u.strip() for u in _ep_uris_str.split(",") if u.strip()]
            if len(_ep_list) >= 2:
                # Detect MPI rank from the launcher environment (OpenMPI, MPICH, Slurm, MV2).
                _rank_str = (
                    os.environ.get("OMPI_COMM_WORLD_RANK")
                    or os.environ.get("PMI_RANK")
                    or os.environ.get("MV2_COMM_WORLD_RANK")
                    or os.environ.get("SLURM_PROCID")
                )
                if _rank_str is not None:
                    _rank = int(_rank_str)
                    _selected = _ep_list[_rank % len(_ep_list)]
                    logging.info(
                        f"Multi-endpoint: rank {_rank} → {_selected} "
                        f"(endpoint {(_rank % len(_ep_list)) + 1} of {len(_ep_list)})"
                    )
                    self.endpoint = _selected
                else:
                    logging.warning(
                        "S3_ENDPOINT_URIS is set but no MPI rank env var found "
                        "(OMPI_COMM_WORLD_RANK / PMI_RANK / MV2_COMM_WORLD_RANK / SLURM_PROCID). "
                        "Using first endpoint for all ranks."
                    )
                    self.endpoint = _ep_list[0]

        _log = logging.getLogger(__name__)
        if _log.isEnabledFor(logging.DEBUG):
            src_key = "storage_options" if storage_options.get("access_key_id") else "AWS_ACCESS_KEY_ID env"
            src_sec = "storage_options" if storage_options.get("secret_access_key") else "AWS_SECRET_ACCESS_KEY env"
            src_ep  = "storage_options" if storage_options.get("endpoint_url") else "AWS_ENDPOINT_URL env"
            _log.debug("ObjStoreLibStorage: credentials/endpoint resolved (storage_options → env fallback):")
            _log.debug(f"  access_key_id  = {'<set> [' + src_key + ']' if self.access_key_id else '<MISSING — set AWS_ACCESS_KEY_ID>'}")
            _log.debug(f"  secret_key     = {'<set> [' + src_sec + ']' if self.secret_access_key else '<MISSING — set AWS_SECRET_ACCESS_KEY>'}")
            _log.debug(f"  endpoint_url   = {self.endpoint!r}  [{src_ep}]")
            _log.debug(f"  region         = {self.region!r}")

        # URI scheme for object storage addressing.
        # s3dlio supports multiple schemes: "s3", "az", "gs", "file", etc.
        # minio and s3torchconnector are S3-only so they always use "s3".
        # Set via storage_options.uri_scheme in YAML config — not via env var.
        self.uri_scheme = storage_options.get("uri_scheme", "s3").rstrip(":/")

        # Object key format configuration:
        # - False/"path": Pass path-only keys (e.g., "path/to/object") — default
        # - True/"uri":   Pass full URIs (e.g., "s3://bucket/path/to/object")
        # Set via storage_options.use_full_object_uri in YAML config — not via env var.
        use_full_uri_str = storage_options.get("use_full_object_uri", "false")
        self.use_full_object_uri = use_full_uri_str.lower() in ("true", "1", "yes")

        if self.use_full_object_uri:
            logging.debug(f"ObjStoreLibStorage: object key format: Full URI ({self.uri_scheme}://container/path/object)")
        else:
            logging.debug("ObjStoreLibStorage: object key format: Path-only (path/object)")

        # Set environment variables for libraries that use them
        if self.access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key

        # Dynamically import and initialize the appropriate library
        if storage_library == "s3dlio":
            logging.debug("ObjStoreLibStorage: using s3dlio — zero-copy multi-protocol (20-30 GiB/s)")
            try:
                import s3dlio
                # s3dlio reads AWS_ENDPOINT_URL for custom endpoints (MinIO, VAST, Ceph).
                # AWS_ENDPOINT_URL_S3 is NOT used by s3dlio — must use AWS_ENDPOINT_URL.
                if self.endpoint:
                    os.environ["AWS_ENDPOINT_URL"] = self.endpoint
                    logging.debug(f"s3dlio: set AWS_ENDPOINT_URL={self.endpoint}")

                # Auto-tune the Tokio async runtime thread count (S3DLIO_RT_THREADS).
                #
                # By default s3dlio sets RT threads = max(4, num_cpus), which on a
                # 128-core machine with NP=8 gives 128 RT threads/rank × 8 = 1,024
                # total Tokio threads — all competing for 128 physical cores.  The
                # runtime is saturated but the excess threads add scheduler overhead
                # without increasing throughput.
                #
                # The actual in-flight concurrency is bounded by write_threads (the
                # number of Python threads issuing concurrent PUT/GET calls).  The
                # Tokio RT only needs enough threads to service those callers plus a
                # small multiplier for async task fanout within each operation.
                #
                # Formula:  2 × write_threads, capped at 128.
                #   - "× 2": each write thread may have one in-progress async task
                #     plus one queued, so 2× prevents starvation.
                #   - Cap 128: prevents runaway thread counts on very large machines
                #     when write_threads is set unusually high.
                #
                # The sentinel _S3DLIO_RT_AUTO lets us distinguish:
                #   - "user set this before launching" → respect it (no sentinel)
                #   - "we auto-set it in an ancestor process (e.g. parent mlpstorage
                #     before spawning mpirun)" → re-compute with the now-finalized
                #     write_threads value, which may be higher than what the parent
                #     computed (parent had write_threads=1 before auto-sizing ran).
                _user_set = (
                    "S3DLIO_RT_THREADS" in os.environ
                    and "_S3DLIO_RT_AUTO" not in os.environ
                )
                if _user_set:
                    logging.debug(
                        f"s3dlio: S3DLIO_RT_THREADS={os.environ['S3DLIO_RT_THREADS']} "
                        "(user-provided before launch, not overriding)"
                    )
                else:
                    _write_threads = getattr(self._args, "write_threads", 8)
                    _rt_threads = min(_write_threads * 3 // 2, 128)
                    os.environ["S3DLIO_RT_THREADS"] = str(_rt_threads)
                    os.environ["_S3DLIO_RT_AUTO"] = "1"   # sentinel: auto-set, may be re-computed
                    logging.info(
                        f"s3dlio: auto-set S3DLIO_RT_THREADS={_rt_threads} "
                        f"(1.5 × write_threads={_write_threads}, cap=128). "
                        "Set S3DLIO_RT_THREADS explicitly to override."
                    )

                self.s3_client = None  # Not used for s3dlio
                self._s3dlio = s3dlio

            except ImportError as e:
                raise ImportError(
                    f"s3dlio is not installed. "
                    f"Install with: pip install s3dlio\nError: {e}"
                )
                
        elif storage_library == "s3torchconnector":
            logging.debug("ObjStoreLibStorage: using s3torchconnector — AWS official S3 connector (5-10 GiB/s)")
            if S3Client is None:
                raise ImportError(
                    "s3torchconnector is not installed. "
                    "Install with: pip install s3torchconnector"
                )
            force_path_style_opt = self._args.s3_force_path_style
            if "s3_force_path_style" in storage_options:
                val = storage_options["s3_force_path_style"]
                force_path_style_opt = val if isinstance(val, bool) else str(val).strip().lower() == "true"
                
            max_attempts_opt = self._args.s3_max_attempts
            if "s3_max_attempts" in storage_options:
                try:
                    max_attempts_opt = int(storage_options["s3_max_attempts"])
                except (TypeError, ValueError):
                    max_attempts_opt = self._args.s3_max_attempts
                    
            profile_opt = storage_options.get("s3_profile", None)

            s3_client_config = S3ClientConfig(
                force_path_style=force_path_style_opt,
                max_attempts=max_attempts_opt,
                profile=profile_opt,
            )
            
            self.s3_client = S3Client(
                region=self.region,
                endpoint=self.endpoint,
                s3client_config=s3_client_config,
            )
            
        elif storage_library == "minio":
            logging.debug("ObjStoreLibStorage: using minio — MinIO native SDK (10-15 GiB/s)")
            try:
                secure = storage_options.get("secure", True)
                self.s3_client = MinIOAdapter(
                    endpoint=self.endpoint,
                    access_key=self.access_key_id,
                    secret_key=self.secret_access_key,
                    region=self.region,
                    secure=secure
                )
            except ImportError as e:
                raise ImportError(
                    f"minio is not installed. "
                    f"Install with: pip install minio\nError: {e}"
                )
        else:
            raise ValueError(
                f"Unknown storage_library: {storage_library}. "
                f"Supported: s3dlio, s3torchconnector, minio"
            )

    @dlp.log
    def get_uri(self, id):
        """
        Construct a full object URI from the configured namespace + object key.
        Uses self.uri_scheme so the output is scheme-agnostic:
          s3://container/path/to/object   (uri_scheme="s3")
          az://container/path/to/object   (uri_scheme="az")
          gs://container/path/to/object   (uri_scheme="gs")
          file:///data/path/to/object     (uri_scheme="file")
        """
        # Already a full URI — return as-is regardless of scheme.
        if '://' in str(id):
            return id
        return f"{self.uri_scheme}://{self.namespace.name}/{id.lstrip('/')}"
    
    def _normalize_object_key(self, uri):
        """
        Decompose an object URI into (container, object_key) for the underlying
        storage library.  Accepts any configured uri_scheme.

        Returns: (container_name, object_key)
          If use_full_object_uri=True:  object_key is the full URI as-is
          If use_full_object_uri=False: object_key is the path portion only
        """
        parsed = urlparse(uri)
        if parsed.scheme != self.uri_scheme:
            raise ValueError(
                f"URI scheme '{parsed.scheme}' does not match configured "
                f"uri_scheme '{self.uri_scheme}' (uri={uri})"
            )

        container_name = parsed.netloc
        object_key = uri if self.use_full_object_uri else parsed.path.lstrip('/')
        return container_name, object_key

    @dlp.log
    def create_namespace(self, exist_ok=False):
        return True

    @dlp.log
    def get_namespace(self):
        return self.get_node(self.namespace.name)

    @dlp.log
    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    @dlp.log
    def get_node(self, id=""):
        return super().get_node(self.get_uri(id))

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        id = self.get_uri(id)  # normalize bare path → full URI (e.g. data/unet3d/train → s3://bucket/data/unet3d/train)
        parsed = urlparse(id)
        if parsed.scheme != self.uri_scheme:
            raise ValueError(
                f"URI scheme '{parsed.scheme}' does not match configured "
                f"uri_scheme '{self.uri_scheme}'"
            )

        container = parsed.netloc
        prefix    = parsed.path.lstrip('/')

        if not use_pattern:
            results = self.list_objects(container, prefix)
            return results

        ext = prefix.split('.')[-1]
        if ext != ext.lower():
            raise Exception(f"Unknown file format {ext}")

        # Pattern matching: check both lowercase and uppercase extensions.
        lower_results = self.list_objects(container, prefix)
        upper_prefix  = prefix.replace(ext, ext.upper())
        upper_results = self.list_objects(container, upper_prefix)
        return lower_results + upper_results

    @dlp.log
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    # Threshold above which s3dlio uses MultipartUploadWriter instead of put_bytes.
    # minio-py uses 5 MiB; 16 MiB is a good balance for MinIO with large objects.
    # Override via S3DLIO_MULTIPART_THRESHOLD_MB env var (set before import).
    _MULTIPART_THRESHOLD = int(os.environ.get("S3DLIO_MULTIPART_THRESHOLD_MB", "16")) * 1024 * 1024

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        if self.storage_library == "s3dlio":
            # s3dlio takes a full URI — id is already built by get_uri().
            # Use getbuffer() when possible: it returns a zero-copy memoryview of
            # the BytesIO internal buffer. getvalue() makes an extra full copy.
            if hasattr(data, 'getbuffer'):
                payload = data.getbuffer()   # zero-copy memoryview (BytesIO)
            elif hasattr(data, 'getvalue'):
                payload = data.getvalue()    # fallback: copy (shouldn't normally happen)
            else:
                payload = data               # already bytes/memoryview
            payload_len = len(payload)
            if payload_len >= self._MULTIPART_THRESHOLD:
                # Use MultipartUploadWriter for large objects — sends multiple
                # concurrent UploadPart requests instead of one giant single-part PUT.
                # This is why minio-py is faster for 140 MiB NPZ files.
                logging.debug(f"put_data: s3dlio multipart upload {id} ({payload_len/1024/1024:.1f} MiB, threshold={self._MULTIPART_THRESHOLD//1024//1024} MiB)")
                with self._s3dlio.MultipartUploadWriter.from_uri(id) as writer:
                    writer.write(payload)
            else:
                self._s3dlio.put_bytes(id, payload)
        else:
            # s3torchconnector or minio - use S3Client API
            bucket_name, object_key = self._normalize_object_key(id)
            writer = self.s3_client.put_object(bucket_name, object_key)
            writer.write(data.getvalue() if hasattr(data, 'getvalue') else data)
            writer.close()
        return None

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        logging.debug(f"get_data: lib={self.storage_library} id={id} offset={offset} length={length}")
        if self.storage_library == "s3dlio":
            # Use s3dlio native API:
            #   get_range() for partial reads (server-side range request — saves bandwidth)
            #   get()       for full object reads — returns BytesView (zero-copy Rust buffer)
            if offset is not None and length is not None:
                logging.debug(f"get_data: s3dlio.get_range({id}, offset={offset}, length={length})")
                return self._s3dlio.get_range(id, offset=offset, length=length)
            logging.debug(f"get_data: s3dlio.get({id})")
            result = self._s3dlio.get(id)
            logging.debug(f"get_data: s3dlio.get returned {len(result)} bytes")
            return result
        else:
            # s3torchconnector or minio - use S3Client API
            bucket_name, object_key = self._normalize_object_key(id)

            if offset is not None and length is not None:
                start = offset
                end = offset + length - 1
                reader = self.s3_client.get_object(bucket_name, object_key, start=start, end=end)
            else:
                reader = self.s3_client.get_object(bucket_name, object_key)

            return reader.read()

    @dlp.log
    def list_objects(self, container_name, prefix=None):
        paths = []
        try:
            if self.storage_library == "s3dlio":
                # Build listing URI with trailing slash so the listing is prefix-scoped.
                key_prefix = prefix.lstrip('/') if prefix else ''
                list_uri = f"{self.uri_scheme}://{container_name}/{key_prefix}".rstrip('/') + '/'
                # recursive=True so nested objects (e.g. train/file.npz) are included.
                full_uris = self._s3dlio.list(list_uri, recursive=True)
                # Strip the full listing URI so returned paths are RELATIVE to the
                # listed prefix — callers expect bare filenames like "file.npz",
                # not bucket-rooted paths like "dlio-train/train/file.npz".
                # NOTE: s3dlio may normalize the URI scheme (e.g. direct:// → file://).
                # Detect the actual returned scheme from the first result so the
                # startswith() prefix strip works regardless of normalization.
                if full_uris and not full_uris[0].startswith(list_uri):
                    actual_scheme = full_uris[0].split('://')[0]
                    strip_prefix = f"{actual_scheme}://{container_name}/{key_prefix}".rstrip('/') + '/'
                else:
                    strip_prefix = list_uri
                for full_uri in full_uris:
                    if full_uri.startswith(strip_prefix):
                        relative = full_uri[len(strip_prefix):]
                        if relative:
                            paths.append(relative)
            else:
                # s3torchconnector / minio: use the S3Client-compatible API.
                if self.use_full_object_uri:
                    p = prefix.lstrip('/') if prefix else ""
                    list_prefix = f"{self.uri_scheme}://{container_name}/{p}"
                else:
                    list_prefix = prefix.lstrip('/') if prefix else ""

                if list_prefix and not list_prefix.endswith('/'):
                    list_prefix += '/'

                obj_stream = self.s3_client.list_objects(container_name, list_prefix)

                for list_obj_result in obj_stream:
                    # Handle both structured results (real libs + MinIOAdapter)
                    # and flat string results (some mocks / alternate implementations).
                    if hasattr(list_obj_result, 'object_info'):
                        items = [obj_info.key for obj_info in list_obj_result.object_info]
                    else:
                        # Flat string — wrap so the loop below is uniform.
                        items = [list_obj_result]

                    for key in items:
                        if list_prefix and key.startswith(list_prefix):
                            paths.append(key[len(list_prefix):])
                        else:
                            paths.append(key)
        except Exception as e:
            print(f"Error listing objects in '{container_name}': {e}")

        return paths

    @dlp.log
    def isfile(self, id):
        return super().isfile(self.get_uri(id))

    def get_basename(self, id):
        return os.path.basename(id)
