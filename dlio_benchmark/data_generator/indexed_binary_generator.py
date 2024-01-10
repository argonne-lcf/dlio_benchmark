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

    def index_file_path_off(self, prefix_path):
        return prefix_path + '.off.idx'

    def index_file_path_size(self, prefix_path):
        return prefix_path + '.sz.idx'

    @dlp.log
    def generate(self):
        """
        Generator for creating data in NPZ format of 3d dataset.
        """
        super().generate()
        np.random.seed(10)
        GB=1073741824
        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim1, dim2 = self.get_dimension()
            sample_size = dim1 * dim2
            total_size = sample_size * self.num_samples
            write_size = total_size
            memory_size = self._args.generation_buffer_size
            if total_size > memory_size:
                write_size = memory_size - (memory_size % sample_size)
            out_path_spec = self.storage.get_uri(self._file_list[i])
            out_path_spec_off_idx = self.index_file_path_off(out_path_spec)
            out_path_spec_sz_idx = self.index_file_path_size(out_path_spec)
            progress(i + 1, self.total_files_to_generate, "Generating Indexed Binary Data")
            prev_out_spec = out_path_spec
            written_bytes = 0
            data_file = open(out_path_spec, "wb")
            off_file = open(out_path_spec_off_idx, "wb")
            sz_file = open(out_path_spec_sz_idx, "wb")
            records = np.random.randint(255, size=write_size, dtype=np.uint8)
            while written_bytes < total_size:
                data_to_write = write_size if written_bytes + write_size <= total_size else total_size - written_bytes
                samples_to_write = data_to_write // sample_size

                # Write data
                myfmt = 'B' * data_to_write
                binary_data = struct.pack(myfmt, *records[:data_to_write])
                data_file.write(binary_data)

                # Write offsets
                myfmt = 'Q' * samples_to_write
                offsets = range(0, data_to_write, sample_size)
                offsets = offsets[:samples_to_write]
                binary_offsets = struct.pack(myfmt, *offsets)
                off_file.write(binary_offsets)

                # Write sizes
                myfmt = 'Q' * samples_to_write
                sample_sizes = [sample_size] * samples_to_write
                binary_sizes = struct.pack(myfmt, *sample_sizes)
                sz_file.write(binary_sizes)

                written_bytes = written_bytes + data_to_write
            data_file.close()
            off_file.close()
            sz_file.close()
        np.random.seed()
