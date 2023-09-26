from time import time
import logging
import math

import tensorflow as tf

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType, Shuffle, FormatType, DatasetType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, Profile

import numpy as np

dlp = Profile(MODULE_DATA_LOADER)


class NativeTFDataLoader(BaseDataLoader):

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.NATIVE_TENSORFLOW)
        self._dataset = None

    @dlp.log
    def read(self):
        read_threads = self._args.read_threads
        if read_threads == 0:
            if self._args.my_rank == 0:
                logging.warning(
                    f"{utcnow()} `read_threads` is set to be 0 for tf.data loader. We change it to 1")
            read_threads = 1

        options = tf.data.Options()
        if "threading" in dir(options):
            options.threading.private_threadpool_size = read_threads
            options.threading.max_intra_op_parallelism = read_threads
        elif "experimental_threading" in dir(options):
            options.experimental_threading.private_threadpool_size = read_threads
            options.experimental_threading.max_intra_op_parallelism = read_threads

        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        self._dataset = ReaderFactory.get_reader(type=self.format_type,
                                                 dataset_type=self.dataset_type,
                                                 thread_index=-1,
                                                 epoch_number=self.epoch_number).read()
        if self._args.prefetch_size > 0:
            self._dataset = self._dataset.prefetch(buffer_size=self._args.prefetch_size)
        self._dataset = self._dataset.batch(batch_size, drop_remainder=True)

    @dlp.log
    def next(self):
        for batch in self._dataset:
            yield batch

    @dlp.log
    def finalize(self):
        pass
