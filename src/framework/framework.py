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

from abc import ABC, abstractmethod
from src.utils.utility import utcnow

from time import sleep
import os
import logging

from mpi4py import MPI
comm = MPI.COMM_WORLD

from src.utils.config import ConfigArguments

class DummyTraceObject(object):
    def __init__(self, string, step, r):
        pass

    def __enter__(self):
        return 1

    def __exit__(self, string, step, r):
        pass


class Framework(ABC):
    def __init__(self):
        self.args = ConfigArguments.get_instance()
        self.output_folder = self.args.output_folder
        self.checkpoint_folder = self.args.checkpoint_folder
        pass

    @abstractmethod
    def init_reader(self, format_type, data_loader=None):
        pass

    @abstractmethod 
    def get_type(self):
        pass
    
    def barrier(self):
        return comm.Barrier()

    def rank(self):
        return comm.rank

    def size(self):
        return comm.size

    @abstractmethod
    def start_framework_profiler(self):
        pass

    @abstractmethod
    def stop_framework_profiler(self):
        pass

    @abstractmethod
    def trace_object(self, string, step, r):
        pass

    def checkpoint(self, epoch, step_number):
        pass

    def model(epoch, epoch_number, step, computation_time):
        sleep(computation_time)

    @abstractmethod
    def compute(self, epoch_number, step, computation_time):
        pass

    @abstractmethod
    def get_reader(self, dataset_type):
        pass

    @abstractmethod
    def is_nativeio_available(self):
        pass
    # Metadata APIs
    def create_node(self, id, exist_ok=False):
        return False

    def get_node(self, id):
        return None

    def walk_node(self, id, use_pattern=False):
        return None

    def delete_node(self, id):
        return False

    # Data APIs
    def put_data(self, id, data, offset=None, length=None):
        return False

    def get_data(self, id, data, offset=None, length=None):
        return None

