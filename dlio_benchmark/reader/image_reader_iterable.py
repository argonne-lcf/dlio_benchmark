"""
JPEG/PNG image reader for local filesystem using parallel prefetch.

Mirrors the structure of image_reader_s3_iterable.py but for local files.
Inherits FormatReader + _LocalFSIterableMixin only — no S3 code paths.

Only the raw byte count is stored per file — no PIL or numpy decode is performed.
"""
# Copyright (c) 2025, UChicago Argonne, LLC. Apache 2.0 License.
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader._local_fs_iterable_mixin import _LocalFSIterableMixin
from dlio_benchmark.utils.utility import utcnow, Profile, dft_ai

dlp = Profile(MODULE_DATA_READER)


class ImageReaderIterable(FormatReader, _LocalFSIterableMixin):
    """
    Parallel-prefetch JPEG/PNG reader for local filesystem.

    All prefetch logic is in _LocalFSIterableMixin.
    PIL/numpy decode is skipped — only raw byte count is kept for telemetry.
    Both dlp and dft_ai are updated (matching ImageReader behaviour).
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._localfs_init()

    @dlp.log
    def open(self, filename):
        super().open(filename)
        return self._local_cache.get(filename, 0)

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        self.logger.debug(f"{utcnow()} sample_index {sample_index}, {self.image_idx}")
        super().get_sample(filename, sample_index)
        byte_count = self._local_cache.get(filename, 0)
        dlp.update(image_size=byte_count)
        dft_ai.update(image_size=byte_count)

    def next(self):
        yield from self._localfs_stream_next()

    @dlp.log
    def read_index(self, image_idx, step):
        filename, _ = self.global_index_map[image_idx]
        self._localfs_ensure_cached(filename)
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
