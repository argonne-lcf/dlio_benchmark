"""
   Copyright (c) 2022, UChicago Argonne, LLC
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

import numpy as np
import struct

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import DataLoaderSampler
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_profiler.logger import fn_interceptor as Profile

dlp = Profile(MODULE_DATA_READER)


class IndexedBinaryReader(FormatReader):
    """
    Reader for Indexed Binary files
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.file_map = {}
        self.load_index()

    def index_file_path_off(self, prefix_path):
        return prefix_path + '.off.idx'

    def index_file_path_size(self, prefix_path):
        return prefix_path + '.sz.idx'

    def read_longs(self, f, n):
        a = np.empty(n, dtype=np.int64)
        f.readinto(a)
        return a

    def load_index_file(self, global_sample_idx, filename, sample_index):
        if filename not in self.file_map:
            offset_file = self.index_file_path_off(filename)
            sz_file = self.index_file_path_size(filename)
            self.file_map[filename] = []
            with open(offset_file, 'rb') as f:
                offsets = self.read_longs(f, self._args.num_samples_per_file)
                logging.debug(f"read offsets {offsets} from file {offset_file}")
                self.file_map[filename].append(offsets)
            with open(sz_file, 'rb') as f:
                sizes = self.read_longs(f, self._args.num_samples_per_file)
                logging.debug(f"read sizes {sizes} from file {sz_file}")
                self.file_map[filename].append(sizes)
    @dlp.log
    def load_index(self):
        if self._args.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            for global_sample_idx, filename, sample_index in self._args.file_map[self.thread_index]:
                self.load_index_file(global_sample_idx, filename, sample_index)
        elif self._args.data_loader_sampler == DataLoaderSampler.INDEX:
            for global_sample_idx, (filename, sample_index) in self._args.global_index_map.items():
                self.load_index_file(global_sample_idx, filename, sample_index)




    @dlp.log
    def open(self, filename):
        super().open(filename)
        return open(filename, "rb")

    @dlp.log
    def close(self, filename):
        super().close(filename)
        self.open_file_map[filename].close()

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        file = self.open_file_map[filename]
        offset = self.file_map[filename][0][sample_index]
        size = self.file_map[filename][1][sample_index]
        logging.debug(f"reading sample from offset {offset} of size {size} from file {filename}")
        file.seek(offset)
        image = np.empty(size, dtype=np.uint8)
        file.readinto(image)
        dlp.update(image_size=size)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()
