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
import math
from numpy import random
from time import time
import pandas as pd

from src.utils.utility import progress, utcnow, perftrace
from src.common.enumerations import Shuffle, FileAccess, ReadType, DatasetType
from src.reader.reader_handler import FormatReader

"""
CSV Reader reader and iterator logic.
"""


class CSVReader(FormatReader):
    def __init__(self, dataset_type, thread_index):
        super().__init__(dataset_type, thread_index)

    @perftrace.event_logging
    def read(self, epoch_number):
        """
        Opens the CSV dataset and reads the rows in memory.
        :param epoch_number: current epoch number
        """
        super().read(epoch_number)
        self._dataset = []
        for file in self._local_file_list:
            val = {
                'file': file,
                'data': None
            }
            if self.read_type == ReadType.IN_MEMORY:
                val['data'] = pd.read_csv(file, compression="infer").to_numpy()
            self._dataset.append(val)
        self.after_read()

    @perftrace.event_logging
    def next(self):
        """
        Iterator for the CSV dataset. In this case, we used the in-memory dataset by sub-setting.
        """
        super().next()
        total = int(math.ceil(self.get_sample_len() / self.batch_size))
        count = 0
        batch = []
        samples_yielded = 0
        for index in range(len(self._dataset)):
            if self.read_type is ReadType.ON_DEMAND or self._dataset[index]["data"] is None:
                self._dataset[index]['data'] = pd.read_csv(self._dataset[index]["file"], compression="infer").to_numpy()
            total_samples = len(self._dataset[index]['data'])
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
                logging.debug(f"{utcnow()} num_set {sample_index} current batch_size {len(batch)}")
                t0 = time()
                my_image = self._dataset[index]['data'][sample_index]
                my_image_resized = np.resize(my_image, (self.max_dimension, self.max_dimension))
                t1 = time()
                perftrace.event_complete(f"CSV_{self.dataset_type}_image_{sample_index}_step_{count}",
                                         "csv_reader..next", t0, t1 - t0)
                logging.debug(f"{utcnow()} new shape of image {my_image_resized.shape}")
                batch.append(my_image_resized)
                is_last = 0 if count < total else 1
                samples_yielded += 1
                if self.samples_per_reader == samples_yielded:
                    is_last = 1
                if is_last:
                    while len(batch) is not self.batch_size:
                        batch.append(np.random.rand(self.max_dimension, self.max_dimension))
                if len(batch) == self.batch_size:
                    count += 1
                    batch = np.array(batch)
                    yield is_last, batch
                    batch = []
                t0 = time()
                if self.samples_per_reader == samples_yielded:
                    break
            if self.read_type is ReadType.ON_DEMAND:
                self._dataset[index]['data'] = None

    @perftrace.event_logging
    def read_index(self, index):
        file_index = math.floor(index / self.num_samples)
        element_index = index % self.num_samples
        if self.read_type is ReadType.ON_DEMAND or self._dataset[file_index]["data"] is None:
            self._dataset[file_index]['data'] = pd.read_csv(self._dataset[file_index]["file"], compression="infer").to_numpy()
        my_image = self._dataset[file_index]['data'][..., element_index]
        logging.debug(f"{utcnow()} shape of image {my_image.shape} self.max_dimension {self.max_dimension}")
        my_image_resized = np.resize(my_image, (self.max_dimension, self.max_dimension))
        logging.debug(f"{utcnow()} new shape of image {my_image.shape}")
        if self.read_type is ReadType.ON_DEMAND:
            self._dataset[index]['data'] = None
        return my_image_resized

    @perftrace.event_logging
    def get_sample_len(self):
        return self.num_samples * len(self._local_file_list)
