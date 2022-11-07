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

from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator
import math
import os

from numpy import random
import csv

from shutil import copyfile
from src.utils.utility import progress
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
        record = random.random((self._dimension * self._dimension))
        records = [record]*self.num_samples
        record_label = 0
        prev_out_spec = ""
        count = 0
        for i in range(0, int(self.total_files_to_generate)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.total_files_to_generate, "Generating CSV Data")
                out_path_spec = self._file_list[i]
                if count == 0:
                    prev_out_spec = out_path_spec
                    df = pd.DataFrame(data=records)
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
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)
