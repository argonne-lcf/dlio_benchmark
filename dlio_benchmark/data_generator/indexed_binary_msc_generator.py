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

import os
import struct
import tempfile

import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import DLIOMPI, Profile, progress

dlp = Profile(MODULE_DATA_GENERATOR)


class IndexedBinaryMscGenerator(DataGenerator):
    """
    Generator for creating Indexed Binary data via the storage abstraction (MSC).

    Unlike IndexedBinaryGenerator, this class does not use MPI collective I/O
    (which requires a shared POSIX filesystem). Each rank independently writes
    its assigned files through self.storage.upload_file().
    """

    def __init__(self):
        super().__init__()

    def index_file_path_off(self, prefix_path):
        return prefix_path + '.off.idx'

    def index_file_path_size(self, prefix_path):
        return prefix_path + '.sz.idx'

    @dlp.log
    def generate(self):
        super().generate()
        np.random.seed(10)
        dim = self.get_dimension(self.total_files_to_generate)

        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim_ = dim[2 * i]
            shape_size = 0
            if isinstance(dim_, list):
                shape_size = np.prod(dim_)
            else:
                dim1 = dim_
                dim2 = dim[2 * i + 1]
                shape_size = dim1 * dim2

            sample_size = shape_size * self._args.record_element_bytes
            total_size = sample_size * self.num_samples
            memory_size = self._args.generation_buffer_size
            write_size = total_size
            if total_size > memory_size:
                write_size = memory_size - (memory_size % sample_size)

            out_path_spec = self._file_list[i]
            out_path_spec_off = self.index_file_path_off(out_path_spec)
            out_path_spec_sz = self.index_file_path_size(out_path_spec)

            progress(i + 1, self.total_files_to_generate, "Generating Indexed Binary Data (MSC)")

            records = np.random.randint(255, size=write_size, dtype=np.uint8)

            tmp_data = tempfile.NamedTemporaryFile(delete=False)
            tmp_off = tempfile.NamedTemporaryFile(delete=False)
            tmp_sz = tempfile.NamedTemporaryFile(delete=False)
            try:
                written_bytes = 0
                while written_bytes < total_size:
                    data_to_write = write_size if written_bytes + write_size <= total_size else total_size - written_bytes
                    samples_to_write = data_to_write // sample_size

                    # Write data
                    myfmt = 'B' * data_to_write
                    binary_data = struct.pack(myfmt, *records[:data_to_write])
                    tmp_data.write(binary_data)
                    struct._clearcache()

                    # Write offsets
                    myfmt = 'Q' * samples_to_write
                    offsets = range(0, data_to_write, sample_size)
                    offsets = offsets[:samples_to_write]
                    binary_offsets = struct.pack(myfmt, *offsets)
                    tmp_off.write(binary_offsets)

                    # Write sizes
                    myfmt = 'Q' * samples_to_write
                    sample_sizes = [sample_size] * samples_to_write
                    binary_sizes = struct.pack(myfmt, *sample_sizes)
                    tmp_sz.write(binary_sizes)

                    written_bytes = written_bytes + data_to_write

                tmp_data.close()
                tmp_off.close()
                tmp_sz.close()

                self.storage.upload_file(out_path_spec, tmp_data.name)
                self.storage.upload_file(out_path_spec_off, tmp_off.name)
                self.storage.upload_file(out_path_spec_sz, tmp_sz.name)
            finally:
                os.unlink(tmp_data.name)
                os.unlink(tmp_off.name)
                os.unlink(tmp_sz.name)

        np.random.seed()
        DLIOMPI.get_instance().comm().Barrier()
