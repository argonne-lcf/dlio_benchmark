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
from time import sleep

import numpy as np
import nvidia
from abc import ABC, abstractmethod
from nvidia import dali

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import DatasetType
from dlio_benchmark.utils.config import ConfigArguments
from dlio_profiler.logger import dlio_logger as PerfTrace, fn_interceptor as Profile

from nvidia.dali import fn
dlp = Profile(MODULE_DATA_READER)

class DaliBaseReader(ABC):

    @dlp.log_init
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self._args = ConfigArguments.get_instance()
        self.batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        self.file_list = self._args.file_list_train if self.dataset_type is DatasetType.TRAIN else self._args.file_list_eval

    @dlp.log
    def _preprocess(self, dataset):
        if self._args.preprocess_time != 0. or self._args.preprocess_time_stdev != 0.:
            t = np.random.normal(self._args.preprocess_time, self._args.preprocess_time_stdev)
            sleep(max(t, 0.0))
        return dataset

    @dlp.log
    def _resize(self, dataset):
        return nvidia.dali.fn.reshape(dataset, shape=[self._args.max_dimension, self._args.max_dimension])

    @abstractmethod
    def _load(self):
        pass

    @dlp.log
    def read(self):
        dataset = self._load()
        #dataset = self._resize(dataset)
        #dataset = nvidia.dali.fn.python_function(dataset, function= self._preprocess, num_outputs=1)
        return dataset

    @abstractmethod
    def finalize(self):
        pass
