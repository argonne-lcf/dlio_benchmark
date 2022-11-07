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

from src.common.enumerations import FrameworkType, Shuffle, FileAccess
from src.framework.framework_factory import FrameworkFactory
from src.utils.utility import utcnow
from src.utils.config import ConfigArguments

import os
import math
import logging
from numpy import random


class FormatReader(ABC):
    def __init__(self):
        self._args = ConfigArguments.get_instance()
        self.read_shuffle = self._args.read_shuffle
        self.seed = self._args.seed
        self.seed_change_epoch = self._args.seed_change_epoch
        self.memory_shuffle = self._args.memory_shuffle
        self.shuffle_size = self._args.shuffle_size
        self.data_dir = self._args.data_folder
        self.record_size = self._args.record_length
        self.prefetch = self._args.prefetch
        self.prefetch_size = self._args.prefetch_size
        self.transfer_size = self._args.transfer_size
        self.file_access = self._args.file_access
        self.my_rank = self._args.my_rank
        self.comm_size = self._args.comm_size
        self.eval_enabled = self._args.do_eval
        self.num_files_eval = self._args.num_files_eval
        self.num_files_train = self._args.num_files_train
        self.total_files = self.num_files_train + self.num_files_eval 
        self.num_samples = self._args.num_samples_per_file
        self._dimension = int(math.sqrt(self.record_size / 8))
        # Batch sizes
        self.batch_size_train = self._args.batch_size
        self.batch_size_eval = self._args.batch_size_eval
        self.batch_size = None
        self._local_file_list = None
        self._local_eval_file_list = None
        self._file_list = None
        self._dataset = None
        self._debug = self._args.debug
        self.framework = FrameworkFactory().get_framework(self._args.framework,
                                                          self._args.profiling)
        # We do this here so we keep the same evaluation files every epoch
        if self.eval_enabled:
            # Pick randomly without replacement the indices of the held-out test set (evaluation set)
            self.eval_indices = random.choice(a=range(self.total_files), size=self.num_files_eval, replace=False)

    @abstractmethod
    def read(self, epoch_number, do_eval=False):
        """
            This method creates and stores the lists of files to be read.
            If using TF, it will partition files between ranks.
            For PT, this is done by the DistributedSampler in data_loader_reader.py.
            We also split files depending on if we are in an evaluation phase or not.
        """
        filenames = os.listdir(self.data_dir)
        fullpaths = [os.path.join(self.data_dir, entry) for entry in filenames]

        # Load the list of files to read
        # We use a held-out data set for evaluation
        # Note: this could also be done in __init__() so as not to repeat the work
        if self.eval_enabled and do_eval:
            files = [path for i, path in enumerate(fullpaths) if i in self.eval_indices]
            num_files = self.num_files_eval
            self.batch_size = self.batch_size_eval
        else:
            if self.eval_enabled:
                files = [path for i, path in enumerate(fullpaths) if i not in self.eval_indices]
            else:
                files = fullpaths
            num_files = self.num_files_train
            self.batch_size = self.batch_size_train

        assert len(files) == num_files, f"Expected {num_files} training files but {len(files)} found. Ensure data was generated correctly."
        self._file_list = files
        read_shuffle = True
        if self.read_shuffle == Shuffle.OFF:
            read_shuffle = False

        seed = None
        if read_shuffle:
            seed = self.seed
            if self.seed_change_epoch:
                seed = self.seed + epoch_number

        if seed is not None:
            random.seed(seed)

        if self.framework.get_type() == FrameworkType.TENSORFLOW and FileAccess.MULTI == self.file_access:
            # Partition the files for each rank
            partition_size = int(math.ceil(len(files) / self.comm_size))
            part_start, part_end = (partition_size * self.my_rank, partition_size * ( self.my_rank + 1))
            self._local_file_list = files[part_start:part_end]
            self._local_file_list_size = len(self._local_file_list)

            if read_shuffle:
                random.shuffle(self._local_file_list)

            logging.debug(f"{utcnow()} Rank {self.my_rank} will read {self._local_file_list_size} files: {self._local_file_list}")

        # For PyTorch the file partitioning is done by the DistributedSampler in src/reader/data_loader_reader.py.
        # We could also partition the files here like for TF and not use the sampler.
        # We did it this way just to follow the UNET3D implementation as closely as possible.
        else:
            self._local_file_list = files
            self._local_file_list_size = len(self._local_file_list)

            if read_shuffle:
                random.shuffle(self._local_file_list)

            logging.debug(f"{utcnow()} Rank {self.my_rank} {self._local_file_list_size} local files: {self._local_file_list}")

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
