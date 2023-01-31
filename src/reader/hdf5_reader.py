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

from src.common.enumerations import Shuffle, FileAccess
from src.reader.reader_handler import FormatReader
import h5py
import math
from numpy import random
import numpy as np
from time import sleep

from src.utils.utility import progress

"""
Reader for HDF5 files for training file.
"""
class HDF5Reader(FormatReader):
    def __init__(self, dataset_type):
        super().__init__(dataset_type)

    def read(self, epoch_number):
        """
        Reading the hdf5 dataset. Here we take just take the filename and they are open during iteration
        :param epoch_number: epoch number for training loop
        """
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
            if self.sample_shuffle != Shuffle.OFF:
                if self.sample_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(num_sets)
            for num_set in num_sets:
                with self.framework.trace_object('Read', num_set / self.batch_size, 1):
                    progress(count, total, "Reading HDF5 Data")
                    count += 1
                    images = dataset[num_set * self.batch_size:(num_set + 1) * self.batch_size]
                resized_images = []
                with self.framework.trace_object('Resize', num_set / self.batch_size, 1):
                    for image in images:
                        resized_images.append(np.resize(image,(self._dimension,self._dimension)))
                    sleep(.001)
                yield resized_images
            file_h5.close()
    def finalize(self):
        pass
