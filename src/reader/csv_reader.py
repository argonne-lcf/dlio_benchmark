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

from src.utils.utility import progress, utcnow, PerfTrace,event_logging
from src.common.enumerations import Shuffle, FileAccess, ReadType, DatasetType
from src.reader.reader_handler import FormatReader

"""
CSV Reader reader and iterator logic.
"""

MY_MODULE = "reader"


class CSVReader(FormatReader):

    def __init__(self, dataset_type, thread_index, epoch_number):
        t0 = time()
        super().__init__(dataset_type, thread_index, epoch_number)
        t1 = time()
        PerfTrace.get_instance().event_complete(
            f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0)

    def open(self, filename):
        t0 = time()
        data = pd.read_csv(filename, compression="infer").to_numpy()
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.open.__qualname__}", MY_MODULE, t0, t1 - t0)
        return data

    def close(self, filename):
        t0 = time()
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.close.__qualname__}", MY_MODULE, t0, t1 - t0)
        pass

    def get_sample(self, filename, sample_index):
        t0 = time()
        my_image = self.open_file_map[filename][sample_index]
        t1 = time()
        t2 = time()
        resized_image = np.resize(my_image, (self._args.max_dimension, self._args.max_dimension))
        t3 = time()
        PerfTrace.get_instance().event_complete(
            f"{self.get_sample.__qualname__}.read", MY_MODULE, t0, t1 - t0)
        PerfTrace.get_instance().event_complete(
            f"{self.get_sample.__qualname__}.resize", MY_MODULE, t2, t3 - t2)
        return resized_image

    @event_logging(module=MY_MODULE)
    def next(self):
        for is_last, batch in super().next():
            yield is_last, batch

    @event_logging(module=MY_MODULE)
    def read_index(self, index):
        return super().read_index(index)

    @event_logging(module=MY_MODULE)
    def finalize(self):
        return super().finalize()
