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
JPEG/PNG image reader using parallel prefetch from S3-compatible object storage.
See _s3_iterable_mixin.py for the full design rationale.

Each image file contains exactly one sample. Prefetch fetches the raw encoded bytes
and stores only the byte count — no PIL or numpy decode is performed.
DLIO's FormatReader.next() yields a pre-allocated random tensor regardless of file
contents; only the byte count is needed for the image_size telemetry metric.

Supported libraries (strictly isolated, no cross-library fallback):
  s3dlio           — s3dlio.get_many(), up to 64 parallel requests, O(1) len(BytesView)
  s3torchconnector — S3IterableDataset.from_objects() + sequential reader
  minio            — ThreadPoolExecutor + Minio SDK, pooled TCP connections
"""

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.image_reader import ImageReader
from dlio_benchmark.reader._s3_iterable_mixin import _S3IterableMixin
from dlio_benchmark.utils.utility import Profile, dft_ai, utcnow

dlp = Profile(MODULE_DATA_READER)


class ImageReaderS3Iterable(ImageReader, _S3IterableMixin):
    """
    Parallel-prefetch JPEG/PNG reader for S3-compatible object stores.

    All prefetch, library routing, and byte-counting logic is in _S3IterableMixin.
    This class is a thin adapter connecting the mixin to DLIO's FormatReader chain.

    Images are 1 sample per file. open_file_map[filename] holds the raw byte count
    (int) used only for telemetry. No PIL or numpy decode is performed.

    ImageReader.get_sample() updates both dlp and dft_ai with image_size —
    we replicate both calls here since we cannot call super().get_sample() (it
    would try to call .nbytes on the cached int).
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index, epoch)
        opts = getattr(self._args, "storage_options", {}) or {}
        self._s3_init(opts)
        self.logger.info(
            f"{utcnow()} ImageReaderS3Iterable [{self._storage_library}] "
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
        # Report byte count for both telemetry systems. Do NOT call super() —
        # ImageReader.get_sample() calls open_file_map[filename].nbytes which would
        # fail because open_file_map[filename] is now an int (byte count), not an array.
        byte_count = self._object_cache.get(filename, 0)
        dlp.update(image_size=byte_count)
        dft_ai.update(image_size=byte_count)

    def next(self):
        self._s3_prefetch_all()
        for batch in super().next():
            yield batch

    # Override the local-FS hooks inherited (via ImageReader) from
    # _LocalFsIterableMixin so they are no-ops for the S3 reader.
    # Without these, ImageReader.read_index() and ImageReader.next() try to
    # open() the object URI as a local path (e.g. "direct:///...") and crash.
    def _localfs_ensure_cached(self, filename):
        pass  # S3 reader uses _object_cache, not local FS cache

    def _localfs_prefetch_all(self):
        pass  # S3 reader uses _s3_prefetch_all, not local FS prefetch

    @dlp.log
    def read_index(self, image_idx, step):
        filename, _ = self.global_index_map[image_idx]
        self._s3_ensure_cached(filename)
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

