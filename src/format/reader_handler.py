from abc import ABC, abstractmethod

from src.common.enumerations import Shuffle
from src.utils.argument_parser import ArgumentParser

import os
import math
from numpy import random


class FormatReader(ABC):
    def __init__(self):
        self._arg_parser = ArgumentParser.get_instance()
        self.read_shuffle = self._arg_parser.args.read_shuffle
        self.seed = self._arg_parser.args.seed
        self.seed_change_epoch = self._arg_parser.args.seed_change_epoch
        self.read_shuffle = self._arg_parser.args.read_shuffle
        self.memory_shuffle = self._arg_parser.args.memory_shuffle
        self.shuffle_size = self._arg_parser.args.shuffle_size
        self.data_dir = self._arg_parser.args.data_folder
        self.record_size = self._arg_parser.args.record_length
        self.prefetch = self._arg_parser.args.prefetch
        self.prefetch_size = self._arg_parser.args.prefetch_size
        self.batch_size = self._arg_parser.args.batch_size
        self.transfer_size = self._arg_parser.args.transfer_size
        self._dataset = None
        self._local_file_list = None

    @abstractmethod
    def read(self, epoch_number):
        read_shuffle = True
        if self.read_shuffle == Shuffle.OFF:
            read_shuffle = False
        if read_shuffle:
            seed = self.seed
            if self.seed_change_epoch:
                seed = self.seed + epoch_number
        filenames = os.listdir(self.data_dir)
        files = list()
        # Iterate over all the entries
        for entry in filenames:
            # Create full path
            fullPath = os.path.join(self.data_dir, entry)
            files.append(fullPath)
        partition_size = int(math.ceil(len(files) / self._arg_parser.args.comm_size))
        part_start, part_end = (partition_size * self._arg_parser.args.my_rank, partition_size * (self._arg_parser.args.my_rank + 1))
        self._local_file_list = files[part_start:part_end]
        if seed is not None:
            random.seed(seed)
        random.shuffle( self._local_file_list)

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
