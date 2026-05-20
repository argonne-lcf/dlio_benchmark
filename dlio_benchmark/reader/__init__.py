"""
Per-format reader factories.

Each function selects the S3-iterable or POSIX-iterable implementation based on
the active storage type and library, then constructs and returns the reader.

Special cases (DALI, O_DIRECT) are NOT handled here — they stay in
reader_factory.py because they require entirely different reader classes.

Usage in reader_factory.py after checking DALI / odirect:
    from dlio_benchmark.reader import create_npz_reader
    return create_npz_reader(dataset_type, thread_index, epoch_number)

storage_library routing for local filesystem (StorageType.LOCAL_FS):
  "direct"  → iterable reader with _LocalFSIterableMixin in O_DIRECT mode
               (s3dlio.get_many() + direct:// URIs, bypasses page cache).
               Use this for NVMe benchmarks.
  <unset>   → iterable reader with _LocalFSIterableMixin in buffered mode
               (ThreadPoolExecutor + Python open(), uses page cache).
"""
from dlio_benchmark.common.enumerations import StorageType
from dlio_benchmark.utils.config import ConfigArguments

_S3_TYPES = (StorageType.S3, StorageType.AISTORE)
_S3_LIBS  = ("s3dlio", "s3torchconnector", "minio")


def _storage_info():
    """Return (storage_type, storage_library) from the active config."""
    args = ConfigArguments.get_instance()
    library = (getattr(args, "storage_options", {}) or {}).get("storage_library")
    return args.storage_type, library


# ---------------------------------------------------------------------------
# NPZ
# ---------------------------------------------------------------------------

def create_npz_reader(dataset_type, thread_index, epoch):
    storage_type, library = _storage_info()
    if storage_type in _S3_TYPES:
        if library in _S3_LIBS:
            from dlio_benchmark.reader.npz_reader_s3_iterable import NPZReaderS3Iterable
            return NPZReaderS3Iterable(dataset_type, thread_index, epoch)
        from dlio_benchmark.reader.npz_reader_s3 import NPZReaderS3
        return NPZReaderS3(dataset_type, thread_index, epoch)
    # LOCAL_FS: both "direct" and buffered (default) use NPZReaderIterable.
    # _LocalFSIterableMixin reads storage_library at _localfs_init() time and
    # routes to O_DIRECT (direct://) or buffered (Python open()) accordingly.
    from dlio_benchmark.reader.npz_reader_iterable import NPZReaderIterable
    return NPZReaderIterable(dataset_type, thread_index, epoch)


# ---------------------------------------------------------------------------
# NPY
# ---------------------------------------------------------------------------

def create_npy_reader(dataset_type, thread_index, epoch):
    storage_type, library = _storage_info()
    if storage_type in _S3_TYPES:
        if library in _S3_LIBS:
            from dlio_benchmark.reader.npy_reader_s3_iterable import NPYReaderS3Iterable
            return NPYReaderS3Iterable(dataset_type, thread_index, epoch)
        from dlio_benchmark.reader.npy_reader_s3 import NPYReaderS3
        return NPYReaderS3(dataset_type, thread_index, epoch)
    # LOCAL_FS: both "direct" and buffered use NPYReaderIterable.
    from dlio_benchmark.reader.npy_reader_iterable import NPYReaderIterable
    return NPYReaderIterable(dataset_type, thread_index, epoch)


# ---------------------------------------------------------------------------
# JPEG / PNG
# ---------------------------------------------------------------------------

def create_image_reader(dataset_type, thread_index, epoch):
    storage_type, library = _storage_info()
    if storage_type in _S3_TYPES:
        if library in _S3_LIBS:
            from dlio_benchmark.reader.image_reader_s3_iterable import ImageReaderS3Iterable
            return ImageReaderS3Iterable(dataset_type, thread_index, epoch)
        # Unrecognised library on S3: fall through to local-style reader;
        # it will fail with a clear error when it tries to open an s3:// URI.
        from dlio_benchmark.reader.image_reader_iterable import ImageReaderIterable
        return ImageReaderIterable(dataset_type, thread_index, epoch)
    # LOCAL_FS: both "direct" and buffered use ImageReaderIterable.
    from dlio_benchmark.reader.image_reader_iterable import ImageReaderIterable
    return ImageReaderIterable(dataset_type, thread_index, epoch)
