from time import time
import logging
import math

import tensorflow as tf

from src.common.enumerations import DataLoaderType, Shuffle, FormatType, DatasetType
from src.data_loader.base_data_loader import BaseDataLoader
from src.reader.reader_factory import ReaderFactory
from src.utils.utility import utcnow, PerfTrace, event_logging

MY_MODULE = "data_loader"
profile_args = {}


class TensorflowDataset(tf.data.Dataset):
    @staticmethod
    def _generator(format_type, dataset_type, epoch_number, thread_index):
        global profile_args
        profile_args["epoch"] = epoch_number
        format_type = format_type.decode('ascii')
        dataset_type = dataset_type.decode('ascii')
        logging.debug(f"{utcnow()} format_type {format_type} dataset_type {dataset_type} tensors")
        reader = ReaderFactory.get_reader(type=FormatType.get_enum(format_type),
                                          dataset_type=DatasetType.get_enum(dataset_type),
                                          thread_index=thread_index,
                                          epoch_number=epoch_number)
        count = 1
        t0 = time()
        for is_last, batch in reader.next():
            t1 = time()
            profile_args["step"] = count
            PerfTrace.get_instance().event_complete(f"TensorflowDataset._generator.next", MY_MODULE, t0, t1 - t0,
                                                    arguments=profile_args)
            yield batch
            if is_last == 1:
                logging.debug(f"{utcnow()} reader thread {thread_index} processed {count} batches")
                return
            count += 1
            t0 = time()

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def __new__(cls, format_type, dataset_type, epoch_number, shape, thread_index):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.uint8,
            output_shapes=shape,
            args=(format_type.value, dataset_type.value, epoch_number, thread_index,),
        )
        return dataset


class TFDataLoader(BaseDataLoader):


    def __init__(self, format_type, dataset_type, epoch_number):
        global profile_args
        profile_args["epoch"] = epoch_number
        t0 = time()
        super().__init__(format_type, dataset_type, epoch_number)
        self._dataset = None
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0,
                                                arguments=profile_args)

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def read(self):
        read_threads = self._args.read_threads
        if read_threads == 0:
            if self._args.my_rank == 0:
                logging.warning(
                    f"{utcnow()} `read_threads` is set to be 0 for tf.data loader. We change it to tf.data.AUTOTUNE")
            read_threads = tf.data.AUTOTUNE

        options = tf.data.Options()
        options.threading.private_threadpool_size = read_threads
        options.threading.max_intra_op_parallelism = read_threads

        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        self._dataset = tf.data.Dataset.from_tensor_slices(range(read_threads)).with_options(options)
        self._dataset = self._dataset.interleave(lambda x: TensorflowDataset(self.format_type, self.dataset_type,
                                                                             self.epoch_number, (
                                                                                 batch_size, self._args.max_dimension,
                                                                                 self._args.max_dimension), x),
                                                 cycle_length=read_threads,
                                                 num_parallel_calls=read_threads)
        if self._args.prefetch_size > 0:
            self._dataset = self._dataset.prefetch(buffer_size=self._args.prefetch_size)

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def next(self):
        global profile_args
        super().next()

        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        step = 1
        for batch in self._dataset:
            profile_args["step"] = step
            yield batch
            step += 1
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} read {step - 1} of {total} batches")

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def finalize(self):
        pass
