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
import numpy as np

from src.common.enumerations import Shuffle, FileAccess
from src.reader.reader_handler import FormatReader
import csv
import math

from numpy import random

from src.utils.utility import progress, utcnow
import pandas as pd
import tensorflow as tf

"""
CSV Reader reader and iterator logic.
"""


class CSVReader(FormatReader):
    def __init__(self, dataset_type):
        super().__init__(dataset_type)

    def read(self, epoch_number):
        """
        Opens the CSV dataset and reads the rows in memory.
        :param epoch_number: current epoch number
        """
        super().read(epoch_number)
        packed_array = []
        count = 1
        for file in self._local_file_list:
            progress(count, len(self._local_file_list), "Opening CSV Data")
            count += 1
            rows = pd.read_csv(file, compression="infer").to_numpy()
            packed_array.append({
                'dataset': rows,
                'current_sample': 0,
                'total_samples': len(rows)
            })
        self._dataset = packed_array

    def next(self):
        """
        Iterator for the CSV dataset. In this case, we used the in-memory dataset by sub-setting.
        """
        super().next()
        total = int(math.ceil(self.get_sample_len() / self.batch_size))
        count = 0
        for element in self._dataset:
            current_index = element['current_sample']
            total_samples = element['total_samples']
            if FileAccess.MULTI == self.file_access:
                num_sets = list(range(0, int(math.ceil(total_samples / self.batch_size))))
            else:
                total_samples_per_rank = int(total_samples / self.comm_size)
                part_start, part_end = (int(total_samples_per_rank * self.my_rank / self.batch_size),
                                        int(total_samples_per_rank * (self.my_rank + 1) / self.batch_size))
                num_sets = list(range(part_start, part_end))

            if self.sample_shuffle != Shuffle.OFF:
                if self.sample_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(num_sets)
            for num_set in num_sets:
                count += 1
                images = []
                for i in range(num_set * self.batch_size, (num_set + 1) * self.batch_size):
                    my_image = element['dataset'][i]
                    logging.debug(f"{utcnow()} shape of image {my_image.shape} self.max_dimension {self.max_dimension}")
                    my_image = np.pad(my_image, ((0, self.max_dimension - my_image.shape[0]),
                                                 (0, self.max_dimension - my_image.shape[1])),
                                      mode='constant', constant_values=0)
                    logging.debug(f"{utcnow()} new shape of image {my_image.shape}")
                    images.append(my_image)
                images = np.array(images)
                is_last = 0 if count < total else 1
                logging.debug(
                    f"{utcnow()} loading numpy array for step {num_set} is_last {is_last} shape {images.shape}")
                logging.debug(f"{utcnow()} completed {count} of {total} is_last {is_last} {len(self._dataset)}")
                yield is_last, images

    def read_index(self, index):
        file_index = math.floor(index / self.num_samples)
        element_index = index % self.num_samples
        my_image = self._dataset[file_index]['dataset'][..., element_index]
        logging.info(f"{utcnow()} shape of image {my_image.shape} self.max_dimension {self.max_dimension}")
        my_image = np.pad(my_image, ((0, self.max_dimension - my_image.shape[0]),
                                     (0, self.max_dimension - my_image.shape[1])),
                          mode='constant', constant_values=0)
        logging.info(f"{utcnow()} new shape of image {my_image.shape}")
        return my_image

    def get_sample_len(self):
        total_samples = 0
        for element in self._dataset:
            total_samples = total_samples + element['total_samples']
        return total_samples
