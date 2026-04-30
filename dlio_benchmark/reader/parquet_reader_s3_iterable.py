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
Parquet reader for S3-compatible object storage using HTTP byte-range GET requests.

Each parquet file may contain many rows (samples) and multiple columns (features).
Reads are row-group-granular: pyarrow.parquet.ParquetFile opens the file by reading
only the footer (a small range request for column and row-group metadata). Individual
row groups are then fetched on demand via server-side Range requests, avoiding full
file downloads.

Supported storage libraries
  s3dlio           — uses s3dlio.get_range(uri, offset, length) and s3dlio.stat(uri)
  s3torchconnector — uses S3Client.get_object() with S3ReaderConstructor.range_based()
                     for native byte-range GETs; object size via HeadObjectResult.
                     No s3dlio dependency. Requires s3torchconnector >= 1.3.0.
  minio            — uses minio.Minio.get_object(bucket, key, offset=, length=)

Memory design
  This reader is a STORAGE BENCHMARK, not a training framework. Row groups are
  fetched to measure I/O throughput, then discarded immediately. Only the
  compressed byte count (an int) is retained per row group so that DLIO's
  telemetry can report image_size. No pyarrow Tables are held in memory between
  calls. Memory per worker process is dominated by framework import overhead
  (~500 MB), not data buffers.

Configuration (under storage_options in the DLIO YAML):
  storage_library:      s3dlio      # or s3torchconnector / minio
  endpoint_url:         http://...  # S3 endpoint; also settable via AWS_ENDPOINT_URL_S3
  columns:              null        # list of column names to read (null = all)

Example YAML snippet:
  dataset:
    format: parquet
    storage_type: s3
    storage_root: my-bucket
    num_samples_per_file: 1024  # must equal actual rows-per-parquet-file
    storage_options:
      storage_library: s3dlio
      endpoint_url: http://127.0.0.1:9000
      columns: ["feature1", "label"]
"""
import bisect
import os
from urllib.parse import urlparse

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


# ── Seekable file-like adapters ───────────────────────────────────────────────


class _S3RangeFile:
    """
    Seekable, readable file-like object backed by s3dlio byte-range GETs.

    Used for s3dlio only. pyarrow.parquet.ParquetFile passes this to its C++
    reader which calls seek/tell/read as needed when scanning column chunks.
    """

    def __init__(self, uri: str):
        self._uri = uri
        self._pos = 0
        self._size = None  # fetched lazily on first seek-from-end or full-read

    def _ensure_size(self):
        if self._size is None:
            import s3dlio
            self._size = s3dlio.stat(self._uri)["size"]

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._ensure_size()
            self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        self._ensure_size()
        remaining = self._size - self._pos
        if remaining <= 0:
            return b""
        if n < 0 or n > remaining:
            n = remaining
        import s3dlio
        data = s3dlio.get_range(self._uri, self._pos, n)
        self._pos += n
        return bytes(data)

    def readall(self) -> bytes:
        return self.read(-1)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self):
        pass


class _MinioRangeFile:
    """
    Seekable, readable file-like object backed by minio byte-range GETs.

    Uses minio.Minio.get_object(bucket, key, offset=offset, length=length)
    for each read() call, matching the s3dlio interface semantics.
    """

    def __init__(self, bucket: str, key: str, client):
        self._bucket = bucket
        self._key = key
        self._client = client
        self._pos = 0
        self._size = None

    def _ensure_size(self):
        if self._size is None:
            self._size = self._client.stat_object(self._bucket, self._key).size

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._ensure_size()
            self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        self._ensure_size()
        remaining = self._size - self._pos
        if remaining <= 0:
            return b""
        if n < 0 or n > remaining:
            n = remaining
        resp = self._client.get_object(
            self._bucket, self._key, offset=self._pos, length=n
        )
        try:
            data = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        self._pos += len(data)
        return data

    def readall(self) -> bytes:
        return self.read(-1)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self):
        pass


# ── Main reader ───────────────────────────────────────────────────────────────


class ParquetReaderS3Iterable(FormatReader):
    """
    Row-group-granular Parquet reader for S3-compatible object storage.

    Opens parquet objects by reading only the footer (column / row-group metadata)
    via a small range request, then fetches individual row groups on demand as
    DLIO requests specific sample indices.

    Memory policy: this is a STORAGE BENCHMARK. After a row group is fetched
    the pyarrow Table is discarded immediately (``del table``). Only the
    compressed byte count (an int) is kept so that ``dlp.update(image_size=N)``
    can report throughput. No Tables are held between calls; per-worker RAM
    overhead is dominated by framework imports (~500 MB), not data.

    DLIO's FormatReader protocol:
      open(filename)               → returns (ParquetFile, cumulative_offsets)
                                     stored in self.open_file_map[filename]
      get_sample(filename, idx)    → looks up the right row group, fetches if
                                     not yet seen, discards Table, updates dlp
      close(filename)              → removes byte-count cache entries for file
      next() / read_index()        → delegate to FormatReader base class

    The cumulative_offsets list has len(num_row_groups + 1) entries; entry i
    is the first global row index of row group i.  Binary search maps a sample
    index to (rg_idx, within-row-group offset) in O(log num_row_groups).
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        args = self._args
        opts = getattr(args, "storage_options", {}) or {}
        # storage_library is REQUIRED — there is no default.  Every object
        # storage workload must explicitly declare which library to use.
        self._storage_library = opts.get("storage_library")
        if self._storage_library is None:
            raise ValueError(
                "storage_options['storage_library'] is required for S3 readers. "
                "Add 'storage_library: <value>' under the 'storage:' section of "
                "your workload YAML.  Supported values: minio, s3dlio, s3torchconnector."
            )
        self._opts = opts
        self._epoch = epoch

        # Optional column selection (list[str] or None = all columns)
        self._columns = opts.get("columns") or None

        # Footer cache: filename → (ParquetFile, cumulative_offsets).
        # Holds ONLY the Parquet footer metadata (schema + row-group offsets),
        # a few KB per file. Keyed by filename so open() never re-reads the
        # footer from S3. Flushed at finalize() (epoch boundary).
        # In ON_DEMAND mode the base class calls open()/close() around every
        # single sample — without this cache that means one footer S3 GET per
        # sample, i.e. 33,000+ wasted GETs per epoch.
        self._pf_cache: dict = {}

        # Row-group byte-count cache: (filename, rg_idx) → int (compressed bytes).
        # Tables are read for I/O measurement then discarded; only the byte count
        # is kept for dlp telemetry. Memory per entry is negligible (~100 bytes).
        # Flushed at finalize() (epoch boundary).
        self._rg_cache: dict = {}

        # s3dlio reads AWS_ENDPOINT_URL_S3 at runtime; set it early if needed.
        if self._storage_library == "s3dlio":
            ep = opts.get("endpoint_url")
            if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
                os.environ["AWS_ENDPOINT_URL_S3"] = ep

        # s3torchconnector: fail immediately if the library is missing or too old.
        # Parquet reading requires byte-range GETs via S3ReaderConstructor.range_based().
        # There is NO silent fallback to s3dlio or any other library.
        self._s3torch_client = None
        if self._storage_library == "s3torchconnector":
            try:
                from s3torchconnector._s3client import (
                    S3Client as _S3TCClient,
                    S3ClientConfig as _S3TCConfig,
                )
                from s3torchconnector import S3ReaderConstructor as _S3TCReaderConstructor
                # Verify range_based() exists — requires s3torchconnector >= 1.3.0
                if not hasattr(_S3TCReaderConstructor, "range_based"):
                    raise RuntimeError(
                        "ParquetReaderS3Iterable: s3torchconnector is too old. "
                        "S3ReaderConstructor.range_based() is required for Parquet "
                        "byte-range reads. Upgrade: pip install --upgrade s3torchconnector"
                    )
            except ImportError as exc:
                raise ImportError(
                    "ParquetReaderS3Iterable: storage_library='s3torchconnector' requires "
                    "the s3torchconnector package (>= 1.3.0). "
                    "Install with: pip install s3torchconnector"
                ) from exc
            ep = opts.get("endpoint_url") or os.environ.get("AWS_ENDPOINT_URL")
            region = opts.get("region", "us-east-1")
            self._s3torch_client = _S3TCClient(
                region=region,
                endpoint=ep or None,
                s3client_config=_S3TCConfig(),
            )
            self.logger.info(
                f"{utcnow()} ParquetReaderS3Iterable: s3torchconnector S3Client ready, "
                f"endpoint={ep!r} region={region!r}"
            )

        # Minio client created lazily once, reused across objects
        self._minio_client = None

        self.logger.info(
            f"{utcnow()} ParquetReaderS3Iterable [{self._storage_library}] "
            f"thread={thread_index} epoch={epoch} "
            f"columns={self._columns}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _uri_for_obj_key(self, obj_key: str) -> str:
        """Return a full URI for a DLIO object key, using the configured uri_scheme."""
        if "://" in obj_key:
            return obj_key
        scheme = self._opts.get("uri_scheme", "s3")
        root = self._args.storage_root.rstrip("/")
        return f"{scheme}://{root}/{obj_key.lstrip('/')}"

    def _uri_for_filename(self, filename: str) -> str:
        """Alias for _uri_for_obj_key for backward compatibility."""
        return self._uri_for_obj_key(filename)

    def _get_minio_client(self):
        if self._minio_client is None:
            from minio import Minio

            opts = self._opts
            endpoint = opts.get("endpoint_url", "")
            if endpoint.startswith("https://"):
                host, secure = endpoint[8:], True
            elif endpoint.startswith("http://"):
                host, secure = endpoint[7:], False
            else:
                host, secure = endpoint, False
            self._minio_client = Minio(
                host,
                access_key=opts.get("access_key_id"),
                secret_key=opts.get("secret_access_key"),
                secure=secure,
                region=opts.get("region", "us-east-1"),
            )
        return self._minio_client

    def _make_range_file(self, filename: str):
        """Create a seekable file-like I/O adapter for the given object key."""
        uri = self._uri_for_obj_key(filename)
        lib = self._storage_library
        if lib == "s3dlio":
            return _S3RangeFile(uri)
        elif lib == "s3torchconnector":
            # Use s3torchconnector's native range-based reader directly.
            # RangedS3Reader (returned by get_object with range_based constructor)
            # is an io.BufferedIOBase that fully supports seek/tell/read/readinto,
            # including SEEK_END — no s3dlio dependency whatsoever.
            from s3torchconnector import S3ReaderConstructor
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            reader_constructor = S3ReaderConstructor.range_based()
            return self._s3torch_client.get_object(
                bucket=bucket, key=key, reader_constructor=reader_constructor
            )
        elif lib == "minio":
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            return _MinioRangeFile(bucket, key, self._get_minio_client())
        else:
            raise ValueError(
                f"ParquetReaderS3Iterable: unknown storage_library {lib!r}; "
                "supported: s3dlio, s3torchconnector, minio"
            )

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """
        Return the (ParquetFile, cumulative_offsets) for this object, reading the
        footer from S3 at most ONCE per epoch.

        The footer is cached in _pf_cache for the lifetime of the epoch. In
        ON_DEMAND mode the base class calls open() before every single sample;
        without this cache that would mean one S3 footer GET per sample.
        """
        if filename in self._pf_cache:
            return self._pf_cache[filename]

        import pyarrow.parquet as pq

        rf = self._make_range_file(filename)
        pf = pq.ParquetFile(rf)
        meta = pf.metadata

        # Build cumulative row offsets [0, rg0_rows, rg0+rg1_rows, ...]
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)

        self.logger.debug(
            f"{utcnow()} ParquetReaderS3Iterable.open {filename} "
            f"row_groups={meta.num_row_groups} total_rows={offsets[-1]}"
        )
        self._pf_cache[filename] = (pf, offsets)
        return self._pf_cache[filename]

    @dlp.log
    def close(self, filename):
        """No-op: footer and byte-count caches are kept for the full epoch.

        In ON_DEMAND mode the base class calls close() after every single sample.
        We must NOT evict either _pf_cache (footer) or _rg_cache (byte counts)
        here — doing so forces a full S3 footer re-read and row-group re-fetch
        for every subsequent sample on the same file.
        Both caches are flushed at epoch boundary in finalize().
        """
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Fetch the row group containing sample_index, record byte count, discard data.

        This is a STORAGE BENCHMARK. The pyarrow Table returned by read_row_group()
        is the I/O we are measuring. It is deleted immediately after the byte count
        is extracted. No row data is held between calls. The DLIO pipeline consumes
        self._args.resized_image (a pre-allocated random tensor) not actual file data.
        """
        pf, offsets = self.open_file_map[filename]

        # Binary search: find rg_idx such that offsets[rg_idx] <= sample_index
        # < offsets[rg_idx + 1].  bisect_right on offsets gives insertion point
        # for sample_index+1, so rg_idx = that - 1, clamped to valid range.
        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        rg_idx = min(rg_idx, pf.metadata.num_row_groups - 1)

        cache_key = (filename, rg_idx)
        if cache_key not in self._rg_cache:
            # Fetch this row group — triggers the range GETs we are benchmarking.
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
