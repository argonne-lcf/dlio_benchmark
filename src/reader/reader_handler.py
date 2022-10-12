from abc import ABC, abstractmethod
from framework.tf_framework import TFFramework

from src.common.enumerations import FrameworkType, Shuffle, FileAccess, Framework
from src.utils.argument_parser import ArgumentParser
from src.framework.framework_factory import FrameworkFactory
from src.utils.utility import utcnow

import os
import math
from numpy import random

import logging



class FormatReader(ABC):
    def __init__(self):
        self._arg_parser = ArgumentParser.get_instance()
        self.read_shuffle = self._arg_parser.args.read_shuffle
        self.seed = self._arg_parser.args.seed
        self.seed_change_epoch = self._arg_parser.args.seed_change_epoch
        self.memory_shuffle = self._arg_parser.args.memory_shuffle
        self.shuffle_size = self._arg_parser.args.shuffle_size
        self.data_dir = self._arg_parser.args.data_folder
        self.record_size = self._arg_parser.args.record_length
        self.prefetch = self._arg_parser.args.prefetch
        self.prefetch_size = self._arg_parser.args.prefetch_size
        self.batch_size = self._arg_parser.args.batch_size
        self.batch_size_eval = self._arg_parser.args.batch_size_eval
        self.transfer_size = self._arg_parser.args.transfer_size
        self.file_access = self._arg_parser.args.file_access
        self.my_rank = self._arg_parser.args.my_rank
        self.comm_size = self._arg_parser.args.comm_size
        self.eval_enabled = self._arg_parser.args.do_eval
        self.num_files_eval = self._arg_parser.args.num_files_eval
        self.num_files_train = self._arg_parser.args.num_files_train
        self.total_files = self.num_files_train + self.num_files_eval 
        self.num_samples = self._arg_parser.args.num_samples
        self._dimension = int(math.sqrt(self.record_size / 8))
        self._local_train_file_list = None
        self._local_eval_file_list = None
        self._dataset = None
        self._debug = self._arg_parser.args.debug
        self.framework = FrameworkFactory().get_framework(self._arg_parser.args.framework,
                                                          self._arg_parser.args.profiling)

        # We do this here so we keep the same evaluation files every epoch
        if self.eval_enabled:
            # Pick randomly without replacement the indices of the held-out test set (evaluation set)
            self.eval_indices = random.choice(a=range(self.total_files), size=self.num_files_eval, replace=False)

    @abstractmethod
    def read(self, epoch_number, do_eval=False):
        """
            This method creates and stores the lists of files to read.
            This is done by separating them between ranks and by training/evaluation phase.
        """
        filenames = os.listdir(self.data_dir)
        fullpaths = [os.path.join(self.data_dir, entry) for entry in filenames]
        if self.eval_enabled:
            # Populate files_train with all training cases
            files_train = [path for i, path in enumerate(fullpaths) if i not in self.eval_indices]
        else:
            files_train = fullpaths
        seed = None

        # Sanity check
        assert len(files_train) == self.num_files_train, f"Expected {self.num_files_train} training files but {len(files_train)} found. Ensure data was generated correctly."

        # Hold out self.num_files_eval files of the dataset to be used for evaluation
        # We only need to do this if we're actually going to read the evaluation set
        if self.eval_enabled and do_eval:
            files_eval = [path for i, path in enumerate(fullpaths) if i in self.eval_indices]
            assert len(files_eval) == self.num_files_eval, f"Expected {self.num_files_eval} eval files but {len(files_eval)} found. Ensure data was generated correctly."

        # For PyTorch, we will split the data files in the data_loader subclass.
        # Same thing if using FileAccess.SHARED e.g. for HDF5 reader
        # Added the get_type() method because isinstance(self.framework, TFFramework) did not return true
        if self.framework.get_type() == FrameworkType.TENSORFLOW and FileAccess.MULTI == self.file_access:

            read_shuffle = True
            if self.read_shuffle == Shuffle.OFF:
                read_shuffle = False
            if read_shuffle:
                seed = self.seed
                if self.seed_change_epoch:
                    seed = self.seed + epoch_number

            if self.eval_enabled and do_eval:
                # Partition the files among rank
                partition_size = int(math.ceil(len(files_eval) / self.comm_size))
                part_start, part_end = (partition_size * self.my_rank, partition_size * ( self.my_rank + 1))
                self._local_eval_file_list = files_eval[part_start:part_end]
                self._local_eval_file_list_size = len(self._local_eval_file_list)

                if seed is not None:
                    random.seed(seed)
                if read_shuffle:
                    random.shuffle(self._local_train_file_list)

                logging.debug(f"{utcnow()} Rank {self.my_rank} will read {self._local_eval_file_list_size} files: {self._local_eval_file_list}")
            else:
                # Here, we calculate how many files each process should read
                # and partition the files for each rank
                partition_size = int(math.ceil(len(files_train) / self.comm_size))
                part_start, part_end = (partition_size * self.my_rank, partition_size * ( self.my_rank + 1))
                self._local_train_file_list = files_train[part_start:part_end]
                self._local_train_file_list_size = len(self._local_train_file_list)

                if seed is not None:
                    random.seed(seed)
                if read_shuffle:
                    random.shuffle(self._local_train_file_list)

                logging.debug(f"{utcnow()} Rank {self.my_rank} will read {self._local_train_file_list_size} files: {self._local_train_file_list}")
        else:
            if self.eval_enabled and do_eval:
                self._local_eval_file_list = files_eval
                self._local_eval_file_list_size = self.num_files_eval
            else:
                self._local_train_file_list = files_train
                self._local_train_file_list_size = self.num_files_train


    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
