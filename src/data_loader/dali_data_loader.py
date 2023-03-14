from time import time
import logging
import math
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.common.enumerations import Shuffle, DataLoaderType, DatasetType
from src.data_loader.base_data_loader import BaseDataLoader
from src.reader.reader_factory import ReaderFactory
from src.utils.utility import utcnow, get_rank, timeit, perftrace

class DaliDataset(object):
    def __init__(self, format_type, dataset_type, epoch_number, num_samples, batch_size, thread_index):
        t0 = time()
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch_number
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=thread_index,
                                               epoch_number=self.epoch_number)
        t1 = time()
        perftrace.event_complete(
            f"DaliDataset_{self.format_type}_{self.dataset_type}_epoch_{self.epoch_number}_init",
            "DaliDataset.init", t0, t1 - t0)
        self.item = self.reader.next()
        self.is_last = 0

    @perftrace.event_logging
    def __iter__(self):
        self.i = 0
        self.n = self.num_samples
        return self

    @perftrace.event_logging
    def __next__(self):
        if self.is_last:
            self.item = self.reader.next()
        t0 = time()
        self.is_last, batch = next(self.item)
        t1 = time()
        perftrace.event_complete(
            f"DaliDataset_{self.format_type}_{self.dataset_type}_epoch_{self.epoch_number}_step_{self.i}",
            "DaliDataset.next", t0, t1 - t0)
        self.i += 1
        labels = [np.uint8([self.i])]*self.batch_size
        return batch, labels

    @perftrace.event_logging
    def __len__(self):
        return self.num_samples

    next = __next__


class DaliDataLoader(BaseDataLoader):

    def __init__(self, format_type, dataset_type):
        super().__init__(format_type, dataset_type)
        self.pipelines = []
        self.epoch_number = None

    @perftrace.event_logging
    def read(self, epoch_number):
        self.epoch_number = epoch_number
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        parallel = True if self._args.read_threads > 0 else False
        self.pipelines = []
        num_threads = 1
        if self._args.read_threads > 0:
            num_threads = self._args.read_threads
        dataset = DaliDataset(self.format_type, self.dataset_type, epoch_number, num_samples, batch_size, 0)
        # None executes pipeline on CPU and the reader does the batching
        pipeline = Pipeline(batch_size=batch_size, num_threads=1, device_id=None, py_num_workers=num_threads)
        with pipeline:
            images, labels = fn.external_source(source=dataset, num_outputs=2, dtype=types.UINT8, parallel=parallel)
            pipeline.set_outputs(images, labels)
        self.pipelines.append(pipeline)
        logging.info(f"{utcnow()} Creating {num_threads} pipelines by {self._args.my_rank} rank ")

    @perftrace.event_logging
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        count = 0
        _dataset = DALIGenericIterator(self.pipelines, ['data', 'label'], size=num_samples)
        t0 = time()
        for batch in _dataset:
            t1 = time()
            perftrace.event_complete(
                f"DaliLoader_{self.format_type}_{self.dataset_type}_epoch_{self.epoch_number}_step_{count}",
                "DaliLoader.next", t0, t1 - t0)
            yield batch
            count += 1
            t0 = time()

    @perftrace.event_logging
    def finalize(self):
        pass
