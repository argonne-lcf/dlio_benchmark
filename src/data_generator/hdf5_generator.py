"""
   Copyright 2021 UChicago Argonne, LLC

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

import h5py
from numpy import random
import math

from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator
from src.utils.utility import progress
from shutil import copyfile

"""
Generator for creating data in HDF5 format.
"""
class HDF5Generator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.chunk_size = self._arg_parser.args.chunk_size
        self.enable_chunking = self._arg_parser.args.enable_chunking

    def generate(self):
        """
        Generate hdf5 data for training. It generates a 3d dataset and writes it to file.
        """
        super().generate()
        samples_per_iter=1024*100
        records = random.random((samples_per_iter, self._dimension, self._dimension))
        record_labels = [0] * self.num_samples
        prev_out_spec = ""
        count = 0
        for i in range(0, int(self.total_files_to_generate)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.total_files_to_generate, "Generating HDF5 Data")
                out_path_spec = self._file_list[i]
                if count == 0:
                    prev_out_spec = out_path_spec
                    hf = h5py.File(out_path_spec, 'w')
                    chunks = None
                    if self.enable_chunking:
                        chunk_dimension = int(math.ceil(math.sqrt(self.chunk_size)))
                        if chunk_dimension > self._dimension:
                            chunk_dimension = self._dimension
                        chunks = (1, chunk_dimension, chunk_dimension)
                    compression = None
                    compression_level = None
                    if self.compression != Compression.NONE:
                        compression = str(self.compression)
                        if self.compression == Compression.GZIP:
                            compression_level = self.compression_level
                    dset = hf.create_dataset('records', (self.num_samples,self._dimension, self._dimension), chunks=chunks, compression=compression,
                                             compression_opts=compression_level)
                    samples_written = 0
                    while samples_written < self.num_samples:
                        if samples_per_iter < self.num_samples-samples_written:
                            samples_to_write = samples_per_iter
                        else:
                            samples_to_write = self.num_samples-samples_written
                        dset[samples_written:samples_written+samples_to_write] = records[:samples_to_write]
                        samples_written += samples_to_write
                    hf.create_dataset('labels', data=record_labels)
                    hf.close()
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)
