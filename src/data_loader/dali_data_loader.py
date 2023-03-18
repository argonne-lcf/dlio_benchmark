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

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.num_samples // self.batch_size:
            # Indicate end of the epoch
            raise StopIteration()
        t0 = time()
        image = self.reader.read_index(sample_idx)
        t1 = time()
        perftrace.event_complete(
            f"DaliDataset_{self.format_type}_{self.dataset_type}_epoch_{self.epoch_number}_step_{sample_info.iteration}",
            "DaliDataset.next", t0, t1 - t0)
        return image, np.uint8([sample_idx])

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
        pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=None, py_num_workers=num_threads)
        with pipeline:
            images, labels = fn.external_source(source=dataset, num_outputs=2, dtype=[types.UINT8, types.UINT8], parallel=parallel, batch=False)
            pipeline.set_outputs(images, labels)
        self.pipelines.append(pipeline)
        logging.info(f"{utcnow()} Creating {num_threads} pipelines by {self._args.my_rank} rank ")

    @perftrace.event_logging
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        for step in range(num_samples//batch_size):
            t0 = time()
            _dataset = DALIGenericIterator(self.pipelines, ['data', 'label'], size=1)
            t1 = time()
            perftrace.event_complete(
                f"DaliLoader.next.iter_{self.format_type}_{self.dataset_type}_epoch_{self.epoch_number}_step_{step}",
                "DaliLoader.next.iter", t0, t1 - t0)
            t0 = time()
            for batch in _dataset:
                t1 = time()
                perftrace.event_complete(
                    f"DaliLoader_{self.format_type}_{self.dataset_type}_epoch_{self.epoch_number}_step_{step}",
                    "DaliLoader.next", t0, t1 - t0)
                yield batch
                t0 = time()

    @perftrace.event_logging
    def finalize(self):
        pass
