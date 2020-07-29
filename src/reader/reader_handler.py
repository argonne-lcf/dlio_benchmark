from abc import ABC, abstractmethod

from src.common.enumerations import Shuffle, FileAccess
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
        self.file_access = self._arg_parser.args.file_access
        self.my_rank = self._arg_parser.args.my_rank
        self.comm_size = self._arg_parser.args.comm_size
        self.num_files = self._arg_parser.args.num_files
        self.num_samples = self._arg_parser.args.num_samples
        self._dataset = None
        self._local_file_list = None

    @abstractmethod
    def read(self, epoch_number):
        filenames = os.listdir(self.data_dir)
        files = list()
        # Iterate over all the entries
        for entry in filenames:
            # Create full path
            fullPath = os.path.join(self.data_dir, entry)
            files.append(fullPath)
        seed = None
        if FileAccess.MULTI == self.file_access:
            files = files[:self.num_files]
            read_shuffle = True
            if self.read_shuffle == Shuffle.OFF:
                read_shuffle = False
            if read_shuffle:
                seed = self.seed
                if self.seed_change_epoch:
                    seed = self.seed + epoch_number
            partition_size = int(math.ceil(len(files) / self.comm_size))
            part_start, part_end = (partition_size * self.my_rank, partition_size * ( self.my_rank + 1))
            self._local_file_list = files[part_start:part_end]
            print("rank {}, file_list {}, size {}".format(self.my_rank, self._local_file_list,partition_size))
            if seed is not None:
                random.seed(seed)
            if read_shuffle:
                random.shuffle(self._local_file_list)
        else:
            self._local_file_list = files

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
