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

import math
import logging
import numpy as np
import tensorflow as tf

from numpy import random
from src.reader.reader_handler import FormatReader
from src.common.enumerations import Shuffle, FileAccess, ReadType

from src.utils.utility import progress, utcnow


class NPZReader(FormatReader):
    """
    Reader for NPZ files
    """

    def __init__(self, dataset_type):
        super().__init__(dataset_type)

    def read(self, epoch_number):
        """
        for each epoch it opens the npz files and reads the data into memory
        :param epoch_number:
        """
        super().read(epoch_number)
        self._dataset = []
        for file in self._local_file_list:
            val = {
                'file': file,
                'data': None
            }
            if self.read_type == ReadType.IN_MEMORY:
                with np.load(file, allow_pickle=True) as data:
                    val['data'] = data["x"]
            self._dataset.append(val)
        self.after_read()

    def next(self):
        """
        The iterator of the dataset just performs memory sub-setting for each portion of the data.
        :return: piece of data for training.
        """
        super().next()
        total = self.total
        count = 0
        batch = []
        for index in range(len(self._dataset)):
            if self._dataset[index]["data"] is None:
                with np.load(self._dataset[index]["file"], allow_pickle=True) as data:
                    self._dataset[index]['data'] = data["x"]
            total_samples = self._dataset[index]['data'].shape[2]
            if FileAccess.MULTI == self.file_access:
                # for multiple file access the whole file would read by each process.
                total_samples_per_rank = total_samples
                sample_index_list = list(range(0, total_samples))
            else:
                total_samples_per_rank = int(total_samples / self.comm_size)
                part_start, part_end = (int(total_samples_per_rank * self.my_rank),
                                        int(total_samples_per_rank * (self.my_rank + 1)))
                sample_index_list = list(range(part_start, part_end))

            if self.sample_shuffle != Shuffle.OFF:
                if self.sample_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(sample_index_list)
            for sample_index in sample_index_list:
                count += 1
                logging.info(f"{utcnow()} num_set {sample_index} current batch_size {len(batch)}")
                my_image = self._dataset[index]['data'][..., sample_index]
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

    def read_index(self, index):
        file_index = math.floor(index / self.num_samples)
        if self._dataset[file_index]["data"] is None:
            with np.load(self._dataset[file_index]["file"], allow_pickle=True) as data:
                self._dataset[file_index]['data'] = data["x"]
        element_index = index % self.num_samples
        my_image = self._dataset[file_index]['data'][..., element_index]
        logging.info(f"{utcnow()} shape of image {my_image.shape} self.max_dimension {self.max_dimension}")
        my_image_resized = np.resize(my_image, (self.max_dimension, self.max_dimension))
        logging.info(f"{utcnow()} new shape of image {my_image_resized.shape}")
        return my_image_resized

    def get_sample_len(self):
        total_samples = 0
        for index in range(len(self._dataset)):
            if self._dataset[index]["data"] is None:
                with np.load(self._dataset[index]["file"], allow_pickle=True) as data:
                    self._dataset[index]['data'] = data["x"]
            total_samples = total_samples + self._dataset[index]['data'].shape[2]
        return total_samples
