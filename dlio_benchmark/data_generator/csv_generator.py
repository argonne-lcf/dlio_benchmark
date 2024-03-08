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
import math
import os

import numpy as np
import csv

from shutil import copyfile
from dlio_benchmark.utils.utility import progress
import pandas as pd

"""
Generator for creating data in CSV format.
"""
class CSVGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
    def generate(self):
        """
        Generate csv data for training. It generates a 2d dataset and writes it to file.
        """
        super().generate()
        np.random.seed(10)
        record_label = 0
        dim = self.get_dimension(self.total_files_to_generate)
        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            progress(i+1, self.total_files_to_generate, "Generating CSV Data")
            dim1 = dim[2*i]
            dim2 = dim[2*i+1]
            record = np.random.randint(255, size=dim1*dim2, dtype=np.uint8)
            records = [record]*self.num_samples
            df = pd.DataFrame(data=records)
            out_path_spec = self.storage.get_uri(self._file_list[i])
            compression = None
            if self.compression != Compression.NONE:
                compression = {
                    "method": str(self.compression)
                }
                if self.compression == Compression.GZIP:
                    out_path_spec = out_path_spec + ".gz"
                elif self.compression == Compression.BZIP2:
                    out_path_spec = out_path_spec + ".bz2"
                elif self.compression == Compression.ZIP:
                    out_path_spec = out_path_spec + ".zip"
                elif self.compression == Compression.XZ:
                    out_path_spec = out_path_spec + ".xz"
            df.to_csv(out_path_spec, compression=compression)
        np.random.seed()
