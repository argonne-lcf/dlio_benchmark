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

from src.common.enumerations import Shuffle, FileAccess
from src.reader.reader_handler import FormatReader
import h5py
import math
from numpy import random

from src.utils.utility import progress


class HDF5Reader(FormatReader):
    def __init__(self):
        super().__init__()

    def read(self, epoch_number):
        super().read(epoch_number)
        packed_array = []
        count = 1
        for file in self._local_file_list:
            progress(count, len(self._local_file_list), "Opening HDF5 Data")
            count += 1
            file_h5 = h5py.File(file, 'r')
            dimention = int(math.sqrt(self.record_size))
            sample = (dimention, dimention)
            dataset_h = file_h5['records']
            current_sample = 0
            packed_array.append({
                'file': file,
                'sample': sample,
                'current_sample': current_sample
            })
            file_h5.close()
        self._dataset = packed_array

    def next(self):
        super().next()
        total = 0
        count = 1
        for element in self._dataset:
            file_h5 = h5py.File(element['file'], 'r')
            dataset = file_h5['records']
            total_samples = dataset.shape[0]
            if FileAccess.MULTI == self.file_access:
                num_sets = list(range(0, int(math.ceil(total_samples/self.batch_size))))
            else:
                total_samples_per_rank = int(total_samples / self.comm_size)
                part_start, part_end = (int(total_samples_per_rank*self.my_rank/self.batch_size), int(total_samples_per_rank*(self.my_rank+1)/self.batch_size))
                num_sets = list(range(part_start, part_end))
            total += len(num_sets)
            if self.memory_shuffle != Shuffle.OFF:
                if self.memory_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(num_sets)
            for num_set in num_sets:
                progress(count, total, "Reading HDF5 Data")
                count += 1
                yield dataset[num_set * self.batch_size:(num_set + 1) * self.batch_size]
            file_h5.close()
    def finalize(self):
        pass
