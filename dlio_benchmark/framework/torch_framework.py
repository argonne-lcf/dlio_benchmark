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

from dlio_benchmark.common.error_code import ErrorCodes
from dlio_benchmark.common.enumerations import FormatType, FrameworkType, DatasetType, DataLoaderType
from dlio_benchmark.data_loader.data_loader_factory import DataLoaderFactory
from dlio_benchmark.framework.framework import Framework, DummyTraceObject
from dlio_benchmark.common.constants import MODULE_AI_FRAMEWORK
import os
import torch
import functools
import logging
from dlio_benchmark.utils.utility import utcnow
from dlio_profiler.logger import fn_interceptor as Profile

from time import sleep, time

from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.storage.storage_factory import StorageFactory

HANDLED_FUNCTIONS = {}
dlp = Profile(MODULE_AI_FRAMEWORK)


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

    @dlp.log_init
    def __init__(self, profiling):
        super().__init__()
        self.profiling = profiling
        self.reader_handler = None

    @dlp.log
    def init_loader(self, format_type, epoch=0, data_loader=None):
        if data_loader is None:
            data_loader = DataLoaderType.PYTORCH
        self.reader_train = DataLoaderFactory.get_loader(data_loader, format_type,
                                                         dataset_type=DatasetType.TRAIN, epoch=epoch)
        self.reader_valid = DataLoaderFactory.get_loader(data_loader, format_type,
                                                         dataset_type=DatasetType.VALID, epoch=epoch)
        self.storage = StorageFactory().get_storage(self.args.storage_type, self.args.storage_root, self.args.framework)

    @dlp.log
    def get_type(self):
        return FrameworkType.PYTORCH

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TorchFramework.__instance is None:
            TorchFramework.__instance = TorchFramework(profiling)
        return TorchFramework.__instance

    @dlp.log
    def start_framework_profiler(self):
        pass

    @dlp.log
    def stop_framework_profiler(self):
        pass

    @dlp.log
    def trace_object(self, string, step, r):
        return DummyTraceObject(string, step, r)

    @dlp.log
    def checkpoint(self, epoch, step_number):
        if self.rank() == 0:
            """
            Performs Checkpointing for a specific step number. It writes different file of different sizes.
            """
            my_rank = self.rank()
            if not self.storage.get_node(self.checkpoint_folder):
                self.storage.create_node(self.checkpoint_folder)

            model_file = os.path.join(self.checkpoint_folder, f"model-{epoch}-{step_number}.bin")

            string_val = "x" * self.args.model_size
            self.storage.put_data(model_file, string_val)

    @dlp.log
    def compute(self, x, epoch_number, step, computation_time):
        torch_sleep(computation_time)

    @dlp.log
    def get_loader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid

    @dlp.log
    def is_nativeio_available(self):
        return False
