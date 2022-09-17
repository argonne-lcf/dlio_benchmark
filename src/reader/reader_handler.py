from abc import ABC, abstractmethod
from framework.tf_framework import TFFramework

from src.common.enumerations import Shuffle, FileAccess
from src.utils.argument_parser import ArgumentParser
from src.framework.framework_factory import FrameworkFactory
from src.utils.utility import utcnow

import os
import math
from numpy import random
from datetime import datetime

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
        self._dataset_train = None
        self._dataset_eval = None
        self.framework = FrameworkFactory().get_framework(self._arg_parser.args.framework,
                                                          self._arg_parser.args.profiling)

        # We do this in the init method instead of read so we keep the same eval case indices for every epoch
        if self.eval_enabled:
            # Pick randomly without replacement the indices of the held-out test set (evaluation set)
            self.eval_indices = random.choice(a=range(self.total_files), size=self.num_files_eval, replace=False)

    @abstractmethod
    def read(self, epoch_number, do_eval=False):
        filenames = os.listdir(self.data_dir)
        fullpaths = [os.path.join(self.data_dir, entry) for entry in filenames]
        # Populate files_train with all training cases
        files_train = [path for i, path in enumerate(fullpaths) if i not in self.eval_indices]
        seed = None

        # Sanity check
        assert len(files_train) == self.num_files_train, f"Expecting to see {self.num_files_train} training files but {len(files_train)} found. Ensure data was generated correctly."

        # Hold out self.num_files_eval files of the dataset to be used for evaluation
        # We only need to do this if we're actually going to read the eval set this epoch
        if self.eval_enabled and do_eval:
            # Populate files_eval with the picked files
            files_eval = [path for i, path in enumerate(fullpaths) if i in self.eval_indices]
            # Sanity check
            assert len(files_eval) == self.num_files_eval, f"Expecting to see {self.num_files_eval} eval files but {len(files_eval)} found. Ensure data was generated correctly."

        # TODO: I think with 1 worker, DLIO will not emulate a single process multi-GPU reading behaviour
        # What would that look like? Maybe we should explicitly call tf.distribute.Strategy and pytorch.DDP
        # Else, we can have multi-process multi-GPU training using horovod, but we have to pin each GPU to a process
        if self.framework is TFFramework and FileAccess.MULTI == self.file_access:

            if self.eval_enabled and do_eval:
                # Here, we calculate how many files each process should read
                # and partition the files for each rank
                partition_size = int(math.ceil(len(files_eval) / self.comm_size))
                part_start, part_end = (partition_size * self.my_rank, partition_size * ( self.my_rank + 1))
                self._local_eval_file_list = files_eval[part_start:part_end]
                self._local_eval_file_list_size = len(self._local_eval_file_list)

                logging.info("{} Rank {} will read {} files: {}".format(utcnow(), self.my_rank, self._local_eval_file_list_size, self._local_eval_file_list))
            else:
                # Here, they used to take a slice of the array up to num_files, i.e.
                # files_train = files_train[:self.num_files]
                # Now that we possibly removed some files for evaluation, we would write 
                # files_train = files_train[:(self.num_files_train)]
                # However, since I got rid of it since in practice we can assume we'll always want to read the whole dataset
                read_shuffle = True
                if self.read_shuffle == Shuffle.OFF:
                    read_shuffle = False
                if read_shuffle:
                    seed = self.seed
                    if self.seed_change_epoch:
                        seed = self.seed + epoch_number
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

                logging.info("{} Rank {} will read {} files: {}".format(utcnow(), self.my_rank, self._local_train_file_list_size, self._local_train_file_list))
        # Else wither the framework is Pytorch and we will do the case file separation in the data_loader_reader class
        # Or we are in FileAccess different than Multi and we also want to do the below
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
