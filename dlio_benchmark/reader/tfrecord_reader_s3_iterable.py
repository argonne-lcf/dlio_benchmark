"""
TFRecord reader using parallel prefetch from S3-compatible object storage.
See _s3_iterable_mixin.py for the full design rationale.

This is a storage benchmark — we measure how fast TFRecord objects can be
fetched from object storage. TFRecord protobuf parsing is pure CPU overhead
that adds noise to the measurement and is never needed: FormatReader.next()
always yields self._args.resized_image, not the actual file contents.

This reader stores only the raw byte count (int) per object, exactly like
NPYReaderS3Iterable, NPZReaderS3Iterable, HDF5ReaderS3Iterable, and
CSVReaderS3Iterable. No tensorflow, no protobuf decoding.

s3dlio's create_dataset() / create_async_loader() fetches raw bytes from
any S3 object regardless of format — TFRecord files are just bytes from
the storage perspective. This reader uses _s3_prefetch_all() (which
dispatches to s3dlio.get_many()) to download all assigned .tfrecord objects
in parallel, storing only the byte count per object.

Three storage libraries are supported (strictly isolated, no cross-library fallback):
  s3dlio           — s3dlio.get_many(), up to 64 parallel requests
  s3torchconnector — S3IterableDataset.from_objects() + sequential reader
  minio            — ThreadPoolExecutor + Minio SDK, pooled TCP connections

Generation (datagen phase) requires tensorflow (TFRecordGenerator) and
framework=tensorflow in the workload config. Reading uses only s3dlio — no
tensorflow required for the I/O measurement.
"""
# Copyright (c) 2025, UChicago Argonne, LLC. Apache 2.0 License.
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader.npy_reader import NPYReader
from dlio_benchmark.reader._s3_iterable_mixin import _S3IterableMixin
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class TFRecordReaderS3Iterable(NPYReader, _S3IterableMixin):
    """
    Parallel-prefetch TFRecord reader for S3-compatible object stores.

    Fetches every assigned .tfrecord object in parallel via s3dlio and stores
    only the raw byte count (int) — no protobuf/tensorflow decoding.
    get_sample() reports that byte count as the image_size telemetry metric.
    The actual I/O transfer is fully measured; the omitted decode step is pure
    CPU overhead irrelevant to storage benchmarking.

    Inherits NPYReader (standard FormatReader harness) rather than TFReader
    because TFReader.next() bypasses the standard open/get_sample/close harness
    and uses tf.data.TFRecordDataset directly. The standard harness calls our
    overridden open()/get_sample()/close() which use _object_cache (byte counts).

    _object_cache[filename] holds an int (byte count), same pattern as all
    other S3 iterable readers.

    Note: read_index() calls FormatReader.read_index() directly to bypass
    NPYReader._localfs_ensure_cached() which would attempt a local filesystem
    read on an S3 URI.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index, epoch)
        opts = getattr(self._args, "storage_options", {}) or {}
        self._s3_init(opts)
        self.logger.info(
            f"{utcnow()} TFRecordReaderS3Iterable [{self._storage_library}] "
            f"thread={thread_index} epoch={epoch}"
        )

    @dlp.log
    def open(self, filename):
        return self._object_cache.get(filename)

    @dlp.log
    def close(self, filename):
        self._object_cache.pop(filename, None)

    @dlp.log
    def get_sample(self, filename, sample_index):
        # Report byte count for telemetry. Do NOT call super() — NPYReader.get_sample()
        # tries to interpret open_file_map[filename] as a numpy array, which would fail
        # because open_file_map[filename] is now an int (byte count).
        dlp.update(image_size=self._object_cache.get(filename, 0))

    def next(self):
        import numpy as np
        if self.thread_index == -1:
            # TFDataLoader TFRECORD mode: single reader over ALL files.
            # file_map is keyed by 0..N-1 thread slots; -1 is never a key.
            all_entries = []
            for entries in self.file_map.values():
                all_entries.extend(entries)
            # Prefetch all unique files via s3dlio in parallel.
            seen = set()
            obj_keys = []
            for _, obj_key, _ in all_entries:
                if obj_key not in seen:
                    seen.add(obj_key)
                    obj_keys.append(obj_key)
            if obj_keys:
                self._object_cache = self._prefetch(obj_keys)
            # Yield batches — same pattern as reader_handler.next().
            batch = []
            total = len(all_entries)
            for i, (_, filename, sample_idx) in enumerate(all_entries):
                self.get_sample(filename, sample_idx)
                batch.append(self._args.resized_image)
                is_last = (i + 1 == total)
                if is_last:
                    while len(batch) < self.batch_size:
                        batch.append(self._args.resized_image)
                if len(batch) == self.batch_size:
                    yield np.array(batch)
                    batch = []
        else:
            self._s3_prefetch_all()
            for batch in super().next():
                yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        filename, _ = self.global_index_map[image_idx]
        self._s3_ensure_cached(filename)
        dlp.update(step=step)
        # Call FormatReader.read_index() directly — skips NPYReader.read_index()
        # which would invoke _localfs_ensure_cached() on an S3 URI and fail.
        return FormatReader.read_index(self, image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
