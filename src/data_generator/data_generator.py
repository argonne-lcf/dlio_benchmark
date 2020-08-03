from abc import ABC, abstractmethod

from src.utils.argument_parser import ArgumentParser
import math
import os
from mpi4py import MPI
from shutil import copyfile


class DataGenerator(ABC):

    def __init__(self):
        self._arg_parser = ArgumentParser.get_instance()
        self.data_dir = self._arg_parser.args.data_folder
        self.record_size = self._arg_parser.args.record_length
        self.file_prefix = self._arg_parser.args.file_prefix
        self.num_files = self._arg_parser.args.num_files
        self.num_samples = self._arg_parser.args.num_samples
        self.my_rank = self._arg_parser.args.my_rank
        self.comm_size = self._arg_parser.args.comm_size
        self.compression = self._arg_parser.args.compression
        self.compression_level = self._arg_parser.args.compression_level
        self._file_prefix = None
        self._dimension = None

    @abstractmethod
    def generate(self):

        if self.my_rank == 0 and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        MPI.COMM_WORLD.barrier()
        self._dimension = int(math.sqrt(self.record_size/8))
        self._file_prefix = os.path.join(self.data_dir, self.file_prefix)
