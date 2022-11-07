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

from abc import ABC, abstractmethod

from src.utils.config import ConfigArguments
import math
import os
from mpi4py import MPI
from shutil import copyfile


class DataGenerator(ABC):

    def __init__(self):
        self._args = ConfigArguments.get_instance()
        self.data_dir = self._args.data_folder
        self.record_size = self._args.record_length
        self.file_prefix = self._args.file_prefix
        self.num_files_train = self._args.num_files_train
        self.do_eval = self._args.do_eval
        self.num_files_eval = self._args.num_files_eval
        self.num_samples = self._args.num_samples_per_file
        self.my_rank = self._args.my_rank
        self.comm_size = self._args.comm_size
        self.compression = self._args.compression
        self.compression_level = self._args.compression_level
        self._file_prefix = None
        self._dimension = None

    @abstractmethod
    def generate(self):

        if self.my_rank == 0 and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        MPI.COMM_WORLD.barrier()
        # What is the logic behind this formula? 
        # Will probably have to adapt to generate non-images
        self._dimension = int(math.sqrt(self.record_size/8))
        self._file_prefix = os.path.join(self.data_dir, self.file_prefix)

        self.total_files_to_generate = self.num_files_train

        if self.num_files_eval > 0:
            self.total_files_to_generate += self.num_files_eval
