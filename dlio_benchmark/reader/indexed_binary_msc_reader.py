"""
   Copyright (c) 2026, UChicago Argonne, LLC
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
import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import DataLoaderSampler
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.utility import Profile, dft_ai

dlp = Profile(MODULE_DATA_READER)


class IndexedBinaryMscReader(FormatReader):
    """
    Reader for Indexed Binary files via MSC.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.storage = StorageFactory().get_storage(
            self._args.storage_type, self._args.storage_root, self._args.framework
        )
        self.file_map_ibr = {}
        self.load_index()

    def index_file_path_off(self, prefix_path):
        prefix_path = prefix_path.replace(self.storage.storage_root, "")
        return prefix_path + '.off.idx'

    def index_file_path_size(self, prefix_path):
        prefix_path = prefix_path.replace(self.storage.storage_root, "")
        return prefix_path + '.sz.idx'

    def binary_file_path(self, prefix_path):
        prefix_path = prefix_path.replace(self.storage.storage_root, "")
        return prefix_path

    def _load_index_array(self, path, dtype=np.uint64):
        """Fetch an entire index file and parse it as a numpy array."""
        raw = self.storage.get_data(path, None)
        return np.frombuffer(raw, dtype=dtype)

    def load_index_file(self, global_sample_idx, filename, sample_index):
        assert isinstance(filename, str), "filename must be a string"
        if filename not in self.file_map_ibr:
            offset_file = self.index_file_path_off(filename)
            sz_file = self.index_file_path_size(filename)
            self.file_map_ibr[filename] = [
                self._load_index_array(offset_file),
                self._load_index_array(sz_file),
            ]
            self.logger.debug(
                f"loaded index for {filename}: "
                f"{len(self.file_map_ibr[filename][0])} offsets, "
                f"{len(self.file_map_ibr[filename][1])} sizes"
            )

    @dlp.log
    def load_index(self):
        if self._args.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            for global_sample_idx, filename, sample_index in self.file_map[self.thread_index]:
                self.load_index_file(global_sample_idx, filename, sample_index)
        elif self._args.data_loader_sampler == DataLoaderSampler.INDEX:
            for global_sample_idx, (filename, sample_index) in self.global_index_map.items():
                self.load_index_file(global_sample_idx, filename, sample_index)

    @dlp.log
    def open(self, filename):
        super().open(filename)
        return self.storage.open(self.binary_file_path(filename))

    @dlp.log
    def close(self, filename):
        super().close(filename)
        self.open_file_map[filename].close()

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        file = self.open_file_map[filename]
        offset = self.file_map_ibr[filename][0][sample_index]
        size = self.file_map_ibr[filename][1][sample_index]
        self.logger.debug(f"reading sample from offset {offset} of size {size} from file {filename}")
        file.seek(offset)
        image = np.empty(size, dtype=np.uint8)
        file.readinto(image)
        dlp.update(image_size=size)

    def next(self):
        for batch in super().next():
            yield batch

    @dft_ai.data.item
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
