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
NPY reader using parallel/streaming fetch from object storage.

NPY files contain a single array (no named key), so decode is simply
np.load(BytesIO(data)) rather than np.load(BytesIO(data))['x'].

Supported libraries:
  s3dlio           — uses s3dlio.get_many() (parallel, up to 64 in-flight requests)
  s3torchconnector — uses S3IterableDataset.from_objects() with sequential reader
                     (single streaming GET per file via s3torchconnector's own API;
                     no s3dlio dependency)
  minio            — uses concurrent.futures.ThreadPoolExecutor with Minio SDK

All objects assigned to this DLIO thread are prefetched before iteration begins.
Note: listing is handled by ObjStoreLibStorage.list_objects(), which dispatches
per library — each library (s3dlio, s3torchconnector, minio) handles its own
listing independently. Delete is not yet implemented for object storage (no-op).

Each library is STRICTLY isolated — there is NO silent fallback to another
library. Configuring a library that is not installed raises ImportError immediately
at construction time, not later during I/O.

NOTE ON INHERITANCE
-------------------
This class inherits FormatReader + _S3IterableMixin ONLY.  It deliberately does
NOT inherit NPYReader (the local-filesystem reader), which carries
_LocalFSIterableMixin.  Mixing that in would cause read_index to call
_localfs_ensure_cached() — which calls open('...') treating an object URI
as a local path.  Keep these two hierarchies strictly separate:

  Object storage:  NPYReaderS3Iterable(FormatReader, _S3IterableMixin)
  Local filesystem: NPYReaderIterable(FormatReader, _LocalFSIterableMixin)
"""
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader._s3_iterable_mixin import _S3IterableMixin
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class NPYReaderS3Iterable(FormatReader, _S3IterableMixin):
    """
    Parallel-prefetch NPY reader for S3-compatible object stores.

    All prefetch, library routing, and byte-counting logic is in _S3IterableMixin.
    This class is a thin adapter connecting the mixin to DLIO's FormatReader chain.
    No local filesystem code is reachable from this class.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        opts = getattr(self._args, "storage_options", {}) or {}
        self._s3_init(opts)
        self.logger.info(
            f"{utcnow()} NPYReaderS3Iterable [{self._storage_library}] "
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
        dlp.update(image_size=self._object_cache.get(filename, 0))

    def next(self):
        yield from self._s3_stream_next()

    @dlp.log
    def read_index(self, image_idx, step):
        filename, _ = self.global_index_map[image_idx]
        self._s3_ensure_cached(filename)
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        self.finalize_s3_bytes()  # report actual bytes → args.record_length
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

