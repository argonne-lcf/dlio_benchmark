"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of DLProfile
 DLIO is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
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
                out_path_spec = "{}_{}_of_{}.csv".format(self._file_prefix, i, self.total_files_to_generate)
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