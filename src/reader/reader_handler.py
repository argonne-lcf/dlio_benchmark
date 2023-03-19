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

from src.common.enumerations import FrameworkType, Shuffle, FileAccess, DatasetType, MetadataType, DataLoaderType, \
    ReadType
from src.framework.framework_factory import FrameworkFactory
from src.storage.storage_factory import StorageFactory
from src.utils.utility import utcnow
from src.utils.config import ConfigArguments
import numpy as np
import os
import math
import logging
from numpy import random
import glob

class FormatReader(ABC):
    def __init__(self, dataset_type, thread_index, epoch_number):
        self.profile_args = {}
        self.profile_args["epoch"] = epoch_number
        self.thread_index = thread_index
        self._args = ConfigArguments.get_instance()
        logging.debug(
            f"Loading {self.__class__.__qualname__} reader on thread {self.thread_index} from rank {self._args.my_rank}")
        self.dataset_type = dataset_type
        self.open_file_map = {}
        self.epoch_number = epoch_number

    @abstractmethod
    def open(self, filename):
        self.profile_args = {}
        self.profile_args["epoch"] = self.epoch_number

    @abstractmethod
    def close(self, filename):
        pass

    @abstractmethod
    def get_sample(self, filename, sample_index):
        self.profile_args["image_idx"] = sample_index
        return

    @abstractmethod
    def next(self):
        random_image = np.random.rand(self._args.max_dimension, self._args.max_dimension)
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        batch = []
        image_processed = 0
        batches_processed = 1
        total_images = len(self._args.file_map[self.thread_index])
        logging.debug(f"{utcnow()} Reading {total_images} images thread {self.thread_index} rank {self._args.my_rank}")

        for filename, sample_index in self._args.file_map[self.thread_index]:
            self.profile_args["step"] = batches_processed
            if filename not in self.open_file_map:
                self.open_file_map[filename] = self.open(filename)
            image = self.get_sample(filename, sample_index)
            batch.append(image)
            image_processed += 1
            is_last = 0 if image_processed < total_images else 1
            if is_last:
                while len(batch) is not batch_size:
                    batch.append(random_image)
            if len(batch) == batch_size:
                batches_processed += 1
                batch = np.array(batch)
                yield is_last, batch
                batch = []
        for filename, sample_index in self._args.file_map:
            if filename in self.open_file_map:
                self.close(filename)
                self.open_file_map[filename] = None

    @abstractmethod
    def read_index(self, index):
        self.profile_args["image_idx"] = index
        filename, sample_index = self._args.global_index_map[index]
        logging.debug(f"{utcnow()} read_index {filename}, {sample_index}")

        if self._args.read_type is ReadType.ON_DEMAND or filename not in self.open_file_map:
            self.open_file_map[filename] = self.open(filename)
        image = self.get_sample(filename, sample_index)
        if self._args.read_type is ReadType.ON_DEMAND:
            self.close(filename)
            self.open_file_map[filename] = None
        return image

    @abstractmethod
    def finalize(self):
        for filename, sample_index in self._args.file_map:
            if filename in self.open_file_map:
                self.close(filename)
                self.open_file_map[filename] = None
