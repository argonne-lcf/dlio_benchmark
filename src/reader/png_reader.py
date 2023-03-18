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

from time import time
import math
import logging
import numpy as np
from PIL import Image

from src.reader.reader_handler import FormatReader
from src.utils.utility import utcnow, PerfTrace,event_logging

MY_MODULE = "reader"
profile_args={}

class PNGReader(FormatReader):
    """
    Reader for PNG files
    """
    classname = "PNGReader"

    def __init__(self, dataset_type, thread_index, epoch_number):
        global profile_args
        profile_args["epoch"] = epoch_number
        t0 = time()
        super().__init__(dataset_type, thread_index, epoch_number)
        t1 = time()
        PerfTrace.get_instance().event_complete(
            f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0, arguments=self.profile_args)

    def open(self, filename):
        t0 = time()
        data = Image.open(filename)
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.open.__qualname__}", MY_MODULE, t0, t1 - t0, arguments=self.profile_args)
        return data

    def close(self, filename):
        t0 = time()
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.close.__qualname__}", MY_MODULE, t0, t1 - t0, arguments=self.profile_args)

    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        t0 = time()
        my_image = self.open_file_map[filename]
        t1 = time()
        t2 = time()
        resized_image = my_image.resize((self._args.max_dimension, self._args.max_dimension))
        t2 = time()
        t3 = time()
        PerfTrace.get_instance().event_complete(
            f"{self.get_sample.__qualname__}.read", MY_MODULE, t0, t1 - t0, arguments=self.profile_args)
        PerfTrace.get_instance().event_complete(
            f"{self.get_sample.__qualname__}.resize", MY_MODULE, t2, t3 - t2, arguments=self.profile_args)
        return np.asarray(resized_image)

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def next(self):
        for is_last, batch in super().next():
            yield is_last, batch

    def read_index(self, index):
        t0 = time()
        val = super().read_index(index)
        t1 = time()
        PerfTrace.get_instance().event_complete(
            f"{self.read_index.__qualname__}", MY_MODULE, t0, t1 - t0, arguments=self.profile_args)
        return val

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def finalize(self):
        return super().finalize()
