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

import numpy as np
from numpy import random

from src.utils.utility import progress
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
        records = random.random((self._dimension, self._dimension, self.num_samples))
        record_labels = [0] * self.num_samples
        prev_out_spec =""
        count = 0
        for i in range(0, int(self.num_files)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.num_files, "Generating NPZ Data")
                out_path_spec = "{}_{}_of_{}.npz".format(self._file_prefix, i, self.num_files)
                if count == 0:
                    prev_out_spec = out_path_spec
                    if self.compression != Compression.ZIP:
                        np.savez(out_path_spec, x=records, y=record_labels)
                    else:
                        np.savez_compressed(out_path_spec, x=records, y=record_labels)
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)