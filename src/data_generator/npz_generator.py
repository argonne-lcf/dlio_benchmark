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

from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator

import logging
import numpy as np
from numpy import random

from src.utils.utility import progress, utcnow
from shutil import copyfile

"""
Generator for creating data in NPZ format.
"""
class NPZGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generator for creating data in NPZ format of 3d dataset.
        """
        super().generate()
        record_labels = [0] * self.num_samples
        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            if (self._dimension_stdev>0):
                dim1, dim2 = [max(int(d), 0) for d in random.normal( self._dimension, self._dimension_stdev, 2)]
            else:
                dim1 = dim2 = self._dimension
            records = random.random((dim1, dim2, self.num_samples))
            out_path_spec = self.storage.get_uri(self._file_list[i])
            progress(i+1, self.total_files_to_generate, "Generating NPZ Data")
            prev_out_spec = out_path_spec
            if self.compression != Compression.ZIP:
                np.savez(out_path_spec, x=records, y=record_labels)
            else:
                np.savez_compressed(out_path_spec, x=records, y=record_labels)
