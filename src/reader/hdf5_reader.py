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

import h5py
import math
from numpy import random
import numpy as np
from time import sleep, time

from src.utils.utility import progress, utcnow, perftrace
from src.common.enumerations import Shuffle, FileAccess, ReadType, DatasetType
from src.reader.reader_handler import FormatReader

"""
Reader for HDF5 files for training file.
"""


class HDF5Reader(FormatReader):
    classname = "HDF5Reader"

    def __init__(self, dataset_type, thread_index, epoch_number):
        t0 = time()
        super().__init__(dataset_type, thread_index, epoch_number)
        t1 = time()
        perftrace.event_complete(
            f"{self.classname}_{self.dataset_type}_init_{self.epoch_number}",
            f"{self.classname}.init", t0, t1 - t0)

    def open(self, filename):
        t0 = time()
        data = h5py.File(filename, 'r')
        t1 = time()
        perftrace.event_complete(f"{self.classname}_{self.dataset_type}_open_{filename}_epoch_{self.epoch_number}",
                                 f"{self.classname}.open", t0, t1 - t0)
        return data

    def close(self, filename):
        t0 = time()
        self.open_file_map[filename].close()
        t1 = time()
        perftrace.event_complete(f"{self.classname}_{self.dataset_type}_close_{filename}_epoch_{self.epoch_number}",
                                 f"{self.classname}.close", t0, t1 - t0)
        pass

    def get_sample(self, filename, sample_index):
        t0 = time()
        my_image = self.open_file_map[filename]['records'][sample_index]
        t1 = time()
        t1p = time()
        resized_image = np.resize(my_image, (self._args.max_dimension, self._args.max_dimension))
        t2 = time()
        perftrace.event_complete(
            f"{self.classname}_{self.dataset_type}_get_{filename}_sample_{sample_index}_epoch_{self.epoch_number}",
            f"{self.classname}.get_sample_read", t0, t1 - t0)
        perftrace.event_complete(
            f"{self.classname}_{self.dataset_type}_process_{filename}_sample_{sample_index}_epoch_{self.epoch_number}",
            f"{self.classname}.get_sample_process", t1p, t2 - t1p)
        return resized_image

    @perftrace.event_logging
    def next(self):
        for is_last, batch in super().next():
            yield is_last, batch

    @perftrace.event_logging
    def read_index(self, index):
        return super().read_index(index)

    @perftrace.event_logging
    def finalize(self):
        return super().finalize()
