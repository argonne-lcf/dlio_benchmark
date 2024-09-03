"""
   Copyright (c) 2024, UChicago Argonne, LLC
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

import h5py
import numpy as np
import math # is needed for chunking case.

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress
from dlio_benchmark.utils.utility import Profile
from shutil import copyfile

from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in HDF5 format.
"""
class HDF5Generator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.chunk_size = self._args.chunk_size
        self.enable_chunking = self._args.enable_chunking

    @dlp.log    
    def generate(self):
        """
        Generate hdf5 data for training. It generates a 3d dataset and writes it to file.
        """
        super().generate()
        np.random.seed(10)
        record_labels = [0] * self.num_samples
        dim = self.get_dimension(self.total_files_to_generate)
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
        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim1 = dim[2*i]
            dim2 = dim[2*i+1]
            records = np.random.randint(255, size=(dim1, dim2, self.num_samples), dtype=np.uint8)
            out_path_spec = self.storage.get_uri(self._file_list[i])
            progress(i+1, self.total_files_to_generate, "Generating NPZ Data")
            hf = h5py.File(out_path_spec, 'w')
            hf.create_dataset('records', (self.num_samples, dim1, dim2), chunks=chunks, compression=compression,
                                    compression_opts=compression_level, dtype=np.uint8, data=records)
            hf.create_dataset('labels', data=record_labels)
            hf.close()
        np.random.seed()
