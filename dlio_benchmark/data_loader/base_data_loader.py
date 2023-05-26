import math
import os
from abc import ABC, abstractmethod

from numpy import random

from dlio_benchmark.common.enumerations import FileAccess, DatasetType, MetadataType, Shuffle
from dlio_benchmark.framework.framework_factory import FrameworkFactory
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.config import ConfigArguments


class BaseDataLoader(ABC):
    def __init__(self, format_type, dataset_type, epoch_number):
        self._args = ConfigArguments.get_instance()
        self.dataset_type = dataset_type
        self.format_type = format_type
        self.epoch_number = epoch_number

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
