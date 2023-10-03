import math
import os
from abc import ABC, abstractmethod

from numpy import random

from dlio_benchmark.common.enumerations import FileAccess, DatasetType, MetadataType, Shuffle
from dlio_benchmark.framework.framework_factory import FrameworkFactory
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.config import ConfigArguments


class BaseDataLoader(ABC):
    def __init__(self, format_type, dataset_type, epoch_number, data_loader_type):
        self._args = ConfigArguments.get_instance()
        self.dataset_type = dataset_type
        self.format_type = format_type
        self.epoch_number = epoch_number
        self.data_loader_type = data_loader_type
        self.num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        self.batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
