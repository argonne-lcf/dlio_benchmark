import math
import os
from abc import ABC, abstractmethod

from numpy import random

from src.common.enumerations import FileAccess, DatasetType, MetadataType, Shuffle
from src.framework.framework_factory import FrameworkFactory
from src.storage.storage_factory import StorageFactory
from src.utils.config import ConfigArguments


class BaseDataLoader(ABC):
    def __init__(self, format_type, dataset_type):
        self._args = ConfigArguments.get_instance()
        self.dataset_type = dataset_type
        self.format_type = format_type

    @abstractmethod
    def read(self, epoch_number):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
