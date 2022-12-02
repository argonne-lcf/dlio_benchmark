"""
   Copyright © 2022, UChicago Argonne, LLC
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

from src.common.error_code import ErrorCodes
from src.common.enumerations import FormatType, FrameworkType, DatasetType
from src.framework.framework import Framework, DummyTraceObject

import os
import torch
import functools
import logging
from src.utils.utility import utcnow

from time import sleep

from src.reader.reader_factory import ReaderFactory

HANDLED_FUNCTIONS = {}

def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator

# Does this annotation mean that torch.mean will be replaced by torch_sleep?
@implements(torch.mean)
def torch_sleep(sleep_time):
    return sleep(sleep_time)

class TorchFramework(Framework):
    __instance = None

    def __init__(self, profiling):
        super().__init__()
        self.profiling = profiling
        self.reader_handler = None

    def init_reader(self, format_type, data_loader=None):
        self.reader_train = ReaderFactory.get_reader(format_type, data_loader=data_loader, dataset_type=DatasetType.TRAIN)
        self.reader_valid = ReaderFactory.get_reader(format_type, data_loader=data_loader, dataset_type=DatasetType.VALID)

    def get_type(self):
        return FrameworkType.PYTORCH

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TorchFramework.__instance is None:
            TorchFramework.__instance = TorchFramework(profiling)
        return TorchFramework.__instance

    def start_framework_profiler(self):
        pass

    def stop_framework_profiler(self):
        pass

    def trace_object(self, string, step, r):
        return DummyTraceObject(string, step, r)

    def checkpoint(self, epoch, step_number):
        if self.rank() == 0:
            """
            Performs Checkpointing for a specific step number. It writes different file of different sizes.
            """
            if not os.path.exists(self.checkpoint_folder):
                os.makedirs(self.checkpoint_folder)
            my_rank = self.rank()
            model_file = os.path.join(self.checkpoint_folder, f"model-{epoch}-{step_number}.bin")

            f = open(model_file, "w")
            string_val = "x" * self.args.model_size 
            f.write(string_val)
            f.close()

    def compute(self, epoch_number, step, computation_time):
        torch_sleep(computation_time)

    def get_reader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid
