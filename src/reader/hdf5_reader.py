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
import logging

from src.common.enumerations import Shuffle, FileAccess
from src.reader.reader_handler import FormatReader
import h5py
import math
from numpy import random
import numpy as np
from time import sleep

from src.utils.utility import progress, utcnow

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
        self._dataset = self._local_file_list

    def next(self):
        """
        This method is called during iteration where a dataset is opened and different regions of the dataset are
        yielded to the training loop
        :return: portion of dataset to be used in step.
        """
        super().next()
        total = int(math.ceil(self.get_sample_len() / self.batch_size))
        count = 0
        batch = []
        for filename in self._dataset:
            file_h5 = h5py.File(filename, 'r')
            dataset = file_h5['records']
            logging.info(f"{utcnow()} dataset shape {dataset.shape}")
            total_samples = dataset.shape[0]
            if FileAccess.MULTI == self.file_access:
                # for multiple file access the whole file would read by each process.
                total_samples_per_rank = total_samples
                sample_index_list = list(range(0, total_samples))
            else:
                total_samples_per_rank = int(total_samples / self.comm_size)
                part_start, part_end = (int(total_samples_per_rank * self.my_rank),
                                        int(total_samples_per_rank * (self.my_rank + 1)))
                sample_index_list = list(range(part_start, part_end))
            for sample_index in sample_index_list:
                count += 1
                logging.info(f"{utcnow()} num_set {sample_index} current batch_size {len(batch)}")
                my_image = dataset[sample_index]
                logging.debug(f"{utcnow()} shape of image {my_image.shape} self.max_dimension {self.max_dimension}")
                my_image_resized = np.resize(my_image, (self.max_dimension, self.max_dimension))
                logging.debug(f"{utcnow()} new shape of image {my_image_resized.shape}")
                batch.append(my_image_resized)
                is_last = 0 if count < total else 1
                if is_last:
                    while len(batch) is not self.batch_size:
                        batch.append(np.random.rand(self.max_dimension, self.max_dimension))
                if len(batch) == self.batch_size:
                    batch = np.array(batch)
                    yield is_last, batch
                    batch = []
            file_h5.close()

    def read_index(self, index):
        file_index = math.floor(index / self.num_samples)
        element_index = index % self.num_samples
        file_h5 = h5py.File(self._dataset[file_index], 'r')
        dataset = file_h5['records']
        my_image = dataset[:][:][element_index:element_index + 1]
        my_image_resized = np.resize(my_image, (self.max_dimension, self.max_dimension))
        logging.debug(f"{utcnow()} new shape of image {my_image_resized.shape}")
        file_h5.close()
        return my_image

    def get_sample_len(self):
        total_samples = 0
        for element in self._dataset:
            file_h5 = h5py.File(element, 'r')
            dataset = file_h5['records']
            total_samples = total_samples + dataset.shape[0]
        return total_samples
