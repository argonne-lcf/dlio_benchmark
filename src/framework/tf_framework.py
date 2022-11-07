"""
   Copyright 2021 UChicago Argonne, LLC

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

import os
import logging

from src.utils.utility import utcnow
from src.common.error_code import ErrorCodes
from src.framework.framework import Framework
from src.reader.reader_factory import ReaderFactory
from src.profiler.profiler_factory import ProfilerFactory
from src.common.enumerations import FrameworkType, Profiler, FormatType, DatasetType

import tensorflow as tf

print(tf.sysconfig.get_link_flags())
import horovod.tensorflow as hvd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
hvd.init()

class TFFramework(Framework):
    __instance = None

    def __init__(self, profiling):
        super().__init__()
        self.profiling = profiling
        if profiling:
            self.tensorboard = ProfilerFactory.get_profiler(Profiler.TENSORBOARD)
        self.reader_handler = None

    def init_reader(self, format_type, data_loader=None):
        self.reader_train = ReaderFactory.get_reader(format_type, data_loader=data_loader, dataset_type=DatasetType.TRAIN)
        self.reader_valid = ReaderFactory.get_reader(format_type, data_loader=data_loader, dataset_type=DatasetType.VALID)

    def get_type(self):
        return FrameworkType.TENSORFLOW

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TFFramework.__instance is None:
            TFFramework.__instance = TFFramework(profiling)
        return TFFramework.__instance

    def barrier(self):
        """
        Barrier implementation using horovod's all-reduce
        """
        const = tf.constant(1)
        reduced = hvd.allreduce(const)

    def rank(self):
        return hvd.rank()

    def size(self):
        return hvd.size()

    def start_framework_profiler(self):
        if self.profiling:
            self.tensorboard.start()

    def stop_framework_profiler(self):
        if self.profiling:
            self.tensorboard.stop()

    def trace_object(self, string, step, r):
        return tf.profiler.experimental.Trace(string, step_num=step, _r=r)

    def checkpoint(self, step_number):
        """
        Performs Checkpointing for a specific step number. It writes different file of different sizes.
        """
        if self.rank() == 0:
            my_rank = self.rank()
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            model_file = os.path.join(self.output_folder, f"model_{step_number}_{my_rank}.bin")
            meta_file = os.path.join(self.output_folder, f"meta_{step_number}_{my_rank}.bin")
            index_file = os.path.join(self.output_folder, f"index_{step_number}_{my_rank}.bin")

            f = open(model_file, "w")
            string_val = "x" * self.args.model_size 
            f.write(string_val)
            f.close()
            # Should these scale with the model size?
            f = open(index_file, "w")
            string_val = "x" * (17371)
            f.write(string_val)
            f.close()
            f = open(meta_file, "w")
            string_val = "x" * (24740228)
            f.write(string_val)
            f.close()

    def compute(self, epoch_number, step, computation_time):
        tf.function(self.model)(epoch_number, step, computation_time)

    def get_reader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid
        
