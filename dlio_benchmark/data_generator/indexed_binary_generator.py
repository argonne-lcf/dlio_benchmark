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

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator

import logging
import numpy as np

from dlio_benchmark.utils.utility import progress, utcnow
from dlio_profiler.logger import fn_interceptor as Profile
from shutil import copyfile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR
import struct

dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in NPZ format.
"""
class IndexedBinaryGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def index_file_path(self, prefix_path):
        return prefix_path + '.idx'

    @dlp.log
    def generate(self):
        """
        Generator for creating data in NPZ format of 3d dataset.
        """
        super().generate()
        np.random.seed(10)
        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim1, dim2 = self.get_dimension()
            sample_size = dim1 * dim2
            total_size = sample_size * self.num_samples
            records = np.random.randint(255, size=total_size, dtype=np.uint8)
            out_path_spec = self.storage.get_uri(self._file_list[i])
            out_path_spec_idx = self.index_file_path(out_path_spec)
            progress(i+1, self.total_files_to_generate, "Generating Indexed Binary Data")
            prev_out_spec = out_path_spec
            myfmt = 'B' * total_size
            binary_data = struct.pack(myfmt, *records)
            with open(out_path_spec, "wb") as f:
                f.write(binary_data)
            myfmt = 'Q' * self.num_samples
            offsets = range(0, total_size, sample_size)
            offsets = offsets[:self.num_samples]
            sample_sizes = [sample_size] * self.num_samples
            binary_offsets = struct.pack(myfmt, *offsets)
            binary_sizes = struct.pack(myfmt, *sample_sizes)
            with open(out_path_spec_idx, "wb") as f:
                f.write(binary_offsets)
                f.write(binary_sizes)
        np.random.seed()
