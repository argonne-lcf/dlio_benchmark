from time import time
import logging
import math

import tensorflow as tf

from src.common.enumerations import DataLoaderType, Shuffle, FormatType, DatasetType
from src.data_loader.base_data_loader import BaseDataLoader
from src.reader.reader_factory import ReaderFactory
from src.utils.utility import utcnow, perftrace



class TensorflowDataset(tf.data.Dataset):
    def _generator(format_type, dataset_type, epoch_number, thread_index):
        format_type = format_type.decode('ascii')
        dataset_type = dataset_type.decode('ascii')
        logging.debug(f"{utcnow()} format_type {format_type} dataset_type {dataset_type} tensors")
        reader = ReaderFactory.get_reader(type=FormatType.get_enum(format_type),
                                          dataset_type=DatasetType.get_enum(dataset_type),
                                          thread_index=thread_index)
        reader.read(epoch_number)
        count = 1
        t0 = time()
        for is_last, batch in reader.next():
            t1 = time()
            perftrace.event_complete(f"TFLoader_{format_type}_{dataset_type}_epoch_{epoch_number}_step_{count}", "TFLoader._generator.next", t0, t1 - t0)
            yield batch
            if is_last == 1:
                logging.debug(f"{utcnow()} reader thread {thread_index} processed {count} batches")
                return
            count += 1
            t0 = time()


    def __new__(cls, format_type, dataset_type, epoch_number, shape, thread_index):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.uint8,
            output_shapes=shape,
            args=(format_type.value, dataset_type.value, epoch_number, thread_index,),
        )
        return dataset


class TFDataLoader(BaseDataLoader):

    def __init__(self, format_type, dataset_type):
        super().__init__(format_type, dataset_type)
        self.read_threads = self._args.read_threads
        self.computation_threads = self._args.computation_threads
        self.format = self._args.format

    @perftrace.event_logging
    def read(self, epoch_number):
        if self.read_threads == 0:
            if self._args.my_rank == 0:
                logging.warning(
                    f"{utcnow()} `read_threads` is set to be 0 for tf.data loader. We change it to tf.data.AUTOTUNE")
            self.read_threads = tf.data.AUTOTUNE

        options = tf.data.Options()
        options.threading.private_threadpool_size = self.read_threads
        options.threading.max_intra_op_parallelism = self.read_threads
        self._dataset = tf.data.Dataset.from_tensor_slices(range(self.read_threads)).with_options(options)
        self._dataset = self._dataset.interleave(lambda x: TensorflowDataset(self.format_type, self.dataset_type,
                                                                 epoch_number, (self.batch_size, self.max_dimension, self.max_dimension), x),
                                     cycle_length=self.read_threads,
                                     num_parallel_calls=self.read_threads)
        if self.sample_shuffle != Shuffle.OFF:
            if self.sample_shuffle == Shuffle.SEED:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size,
                                          seed=self.seed)
            else:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        self._dataset = dataset.batch(self.batch_size, drop_remainder=True)

        if self.prefetch_size > 0:
            self._dataset = self._dataset.prefetch(buffer_size=self.prefetch_size)


    @perftrace.event_logging
    def next(self):
        super().next()

        if self._debug:
            total = math.floor(self.num_samples * len(self._file_list) / self.batch_size / self.comm_size)
            logging.debug(f"{utcnow()} Rank {self.my_rank} should read {total} batches")

        for batch in self._dataset:
            yield batch

    def finalize(self):
        pass