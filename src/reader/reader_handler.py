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

from src.common.enumerations import FrameworkType, Shuffle, FileAccess, DatasetType, MetadataType
from src.framework.framework_factory import FrameworkFactory
from src.storage.storage_factory import StorageFactory
from src.utils.utility import utcnow
from src.utils.config import ConfigArguments

import os
import math
import logging
from numpy import random
import glob


class FormatReader(ABC):
    def __init__(self, dataset_type):
        self._args = ConfigArguments.get_instance()
        self.file_shuffle = self._args.file_shuffle
        self.seed = self._args.seed
        self.seed_change_epoch = self._args.seed_change_epoch
        self.sample_shuffle = self._args.sample_shuffle
        self.shuffle_size = self._args.shuffle_size
        self.data_dir = self._args.data_folder
        self.record_size = self._args.record_length
        self.record_size_stdev = self._args.record_length_stdev
        self.prefetch_size = self._args.prefetch_size
        self.transfer_size = self._args.transfer_size
        
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
        self._file_list_train = None
        self._file_list_eval = None
        self._file_list = None
        self._dataset = None
        self._debug = self._args.debug
        self.dataset_type = dataset_type
        self.framework = FrameworkFactory().get_framework(self._args.framework,
                                                          self._args.do_profiling)
        self.storage = StorageFactory().get_storage(self._args.storage_type, self._args.storage_root, self._args.framework)
        # We do this here so we keep the same evaluation files every epoch
        if self.num_files_train > 1 or self.num_samples == 1:
            self.file_acess = FileAccess.MULTI
        else:
            self.file_acess = FileAccess.SHARED
        if self.dataset_type == DatasetType.TRAIN:
            filenames = self.storage.walk_node(os.path.join(self.data_dir,  "train"))
            if self.storage.get_node(os.path.join(self.data_dir, "train", filenames[0])) == MetadataType.DIRECTORY:
                fullpaths=self.storage.walk_node(os.path.join(self.data_dir, "train/*/*"), use_pattern=True)
            else:
                fullpaths = [self.storage.get_uri(os.path.join(self.data_dir, "train", entry)) for entry in filenames]

            num_files = self.num_files_train
            self.batch_size = self.batch_size_train
            assert len(fullpaths) == num_files, f"Expected {num_files} training files but {len(fullpaths)} found. Ensure data was generated correctly."            
        elif self.dataset_type == DatasetType.VALID:
            filenames = self.storage.walk_node(os.path.join(self.data_dir, "valid/"))
            if (len(filenames)>0):
                if self.storage.get_node(os.path.join(self.data_dir, "valid", filenames[0])) == MetadataType.DIRECTORY:
                    fullpaths=self.storage.walk_node(os.path.join(self.data_dir, "valid/*/*"), use_pattern=True)
                else:
                    fullpaths = [self.storage.get_uri(os.path.join(self.data_dir, "valid", entry)) for entry in filenames]
                num_files = self.num_files_eval
                self.batch_size = self.batch_size_eval
                assert len(fullpaths) == num_files, f"Expected {num_files} validation files but {len(fullpaths)} found. Ensure data was generated correctly."
            else:
                fullpaths=[]

        self._file_list = fullpaths
        self._local_file_list = self._file_list[self.my_rank::self.comm_size]


    @abstractmethod
    def read(self, epoch_number):
        """
            This method creates and stores the lists of files to be read.
            If using TF, it will partition files between ranks.
            For PT, this is done by the DistributedSampler in data_loader_reader.py.
            We also split files depending on if we are in an evaluation phase or not.
        """
        file_shuffle = True
        if self.file_shuffle == Shuffle.OFF:
            file_shuffle = False

        seed = None
        if file_shuffle:
            seed = self.seed
            if self.seed_change_epoch:
                seed = self.seed + epoch_number

        if seed is not None:
            random.seed(seed)

        if file_shuffle:
            random.shuffle(self._file_list)
        self._local_file_list = self._file_list[self.my_rank::self.comm_size]
        
    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
