from time import time
import logging
import math

import tensorflow as tf

from src.common.constants import MODULE_DATA_LOADER
from src.common.enumerations import DataLoaderType, Shuffle, FormatType, DatasetType
from src.data_loader.base_data_loader import BaseDataLoader
from src.reader.reader_factory import ReaderFactory
from src.utils.utility import utcnow, Profile


class TensorflowDataset(tf.data.Dataset):
    @staticmethod
    def _generator(format_type, dataset_type, epoch_number, thread_index):
        with Profile(name="TensorflowDataset._generator", cat=MODULE_DATA_LOADER, epoch=epoch_number):
            format_type = format_type.decode('ascii')
            dataset_type = dataset_type.decode('ascii')
            logging.debug(f"{utcnow()} format_type {format_type} dataset_type {dataset_type} tensors")
            reader = ReaderFactory.get_reader(type=FormatType.get_enum(format_type),
                                              dataset_type=DatasetType.get_enum(dataset_type),
                                              thread_index=thread_index,
                                              epoch_number=epoch_number)
            step = 1
            with Profile(name=f"TensorflowDataset._generator.next", cat=MODULE_DATA_LOADER) as lp:
                for is_last, batch in reader.next():
                    lp.update(epoch=epoch_number, step=step).flush()
                    yield batch
                    if is_last == 1:
                        logging.debug(f"{utcnow()} reader thread {thread_index} processed {step} batches")
                        return
                    step += 1
                    lp.reset()

    def __new__(cls, format_type, dataset_type, epoch_number, shape, thread_index):
        with Profile(name="TensorflowDataset.__new__", cat=MODULE_DATA_LOADER, epoch=epoch_number):
            dataset = tf.data.Dataset.from_generator(
                cls._generator,
                output_types=tf.uint8,
                output_shapes=shape,
                args=(format_type.value, dataset_type.value, epoch_number, thread_index,),
            )
            return dataset


class TFDataLoader(BaseDataLoader):

    def __init__(self, format_type, dataset_type, epoch_number):
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_DATA_LOADER, epoch=epoch_number):
            super().__init__(format_type, dataset_type, epoch_number)
            self._dataset = None

    def read(self):
        with Profile(name=f"{self.read.__qualname__}", cat=MODULE_DATA_LOADER, epoch=self.epoch_number):
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
                                                                                     batch_size,
                                                                                     self._args.max_dimension,
                                                                                     self._args.max_dimension), x),
                                                     cycle_length=read_threads,
                                                     num_parallel_calls=read_threads)
            if self._args.prefetch_size > 0:
                self._dataset = self._dataset.prefetch(buffer_size=self._args.prefetch_size)

    def next(self):
        with Profile(name=f"{self.next.__qualname__}", cat=MODULE_DATA_LOADER, epoch=self.epoch_number):
            super().next()
            total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
            step = 1
            with Profile(name=f"{self.next.__qualname__}.next", cat=MODULE_DATA_LOADER) as lp:
                for batch in self._dataset:
                    lp.update(epoch=self.epoch_number, step=step).flush()
                    yield batch
                    step += 1
                    lp.reset()
            logging.debug(f"{utcnow()} Rank {self._args.my_rank} read {step - 1} of {total} batches")

    def finalize(self):
        with Profile(name=f"{self.finalize.__qualname__}", cat=MODULE_DATA_LOADER, epoch=self.epoch_number):
            pass
