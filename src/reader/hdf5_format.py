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
import numpy as np
import tensorflow as tf
from time import sleep

from src.utils.utility import progress

"""
Reader for HDF5 files for training file.
"""
class HDF5Reader(FormatReader):
    def __init__(self):
        super().__init__()

    def read(self, epoch_number):
        """
        Reading the hdf5 dataset. Here we take just take the filename and they are open during iteration
        :param epoch_number: epoch number for training loop
        """
        super().read(epoch_number)
        packed_array = []
        count = 1
        for file in self._local_train_file_list:
            progress(count, len(self._local_train_file_list), "Opening HDF5 Data")
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
        """
        This method is called during iteration where a dataset is opened and different regions of the dataset are
        yielded to the training loop
        :return: portion of dataset to be used in step.
        """
        super().next()
        total = 0
        count = 1
        for element in self._dataset:
            file_h5 = h5py.File(element['file'], 'r')
            dataset = file_h5['records']
            total_samples = dataset.shape[0]
            if FileAccess.MULTI == self.file_access:
                # for multiple file access the whole file would read by each process.
                num_sets = list(range(0, int(math.ceil(total_samples/self.batch_size))))
            else:
                # for shared file access a part of file would be read by each process.
                total_samples_per_rank = int(total_samples / self.comm_size)
                part_start, part_end = (int(total_samples_per_rank*self.my_rank/self.batch_size), int(total_samples_per_rank*(self.my_rank+1)/self.batch_size))
                num_sets = list(range(part_start, part_end))
            total += len(num_sets)
            if self.memory_shuffle != Shuffle.OFF:
                if self.memory_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(num_sets)
            for num_set in num_sets:
                with tf.profiler.experimental.Trace('Read', step_num=num_set / self.batch_size, _r=1):
                    progress(count, total, "Reading HDF5 Data")
                    count += 1
                    images = dataset[num_set * self.batch_size:(num_set + 1) * self.batch_size]
                resized_images = []
                with tf.profiler.experimental.Trace('Resize', step_num=num_set / self.batch_size, _r=1):
                    for image in images:
                        resized_images.append(np.resize(image,(self._dimension,self._dimension)))
                    sleep(.001)
                yield resized_images
            file_h5.close()
    def finalize(self):
        pass
