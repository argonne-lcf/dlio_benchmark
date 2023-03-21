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
from src.utils.utility import utcnow, get_rank, timeit, PerfTrace, event_logging

MY_MODULE="data_loader"
profile_args = {}

class DaliDataset(object):
    def __init__(self, format_type, dataset_type, epoch_number, num_samples, batch_size, thread_index):
        global profile_args
        profile_args["epoch"] = epoch_number
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
        self.item = self.reader.next()
        self.is_last = 0
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0,
                                                arguments=profile_args)

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.num_samples // self.batch_size:
            # Indicate end of the epoch
            raise StopIteration()
        global profile_args
        profile_args["image_idx"] = sample_idx
        t0 = time()
        image = self.reader.read_index(sample_idx)
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__call__.__qualname__}", MY_MODULE, t0, t1 - t0,
                                                arguments=profile_args)
        return image, np.uint8([sample_idx])

class DaliDataLoader(BaseDataLoader):

    def __init__(self, format_type, dataset_type, epoch_number):
        global profile_args
        profile_args["epoch"] = epoch_number
        t0 = time()
        super().__init__(format_type, dataset_type, epoch_number)
        self.pipelines = []
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0,
                                                arguments=profile_args)

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def read(self):
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        parallel = True if self._args.read_threads > 0 else False
        self.pipelines = []
        num_threads = 1
        if self._args.read_threads > 0:
            num_threads = self._args.read_threads
        dataset = DaliDataset(self.format_type, self.dataset_type, self.epoch_number, num_samples, batch_size, 0)
        # None executes pipeline on CPU and the reader does the batching
        pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=None, py_num_workers=num_threads)
        with pipeline:
            images, labels = fn.external_source(source=dataset, num_outputs=2, dtype=[types.UINT8, types.UINT8], parallel=parallel, batch=False)
            pipeline.set_outputs(images, labels)
        self.pipelines.append(pipeline)
        logging.info(f"{utcnow()} Creating {num_threads} pipelines by {self._args.my_rank} rank ")

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def next(self):
        global profile_args
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        for step in range(num_samples//batch_size):
            profile_args["step"] = step
            t0 = time()
            _dataset = DALIGenericIterator(self.pipelines, ['data', 'label'], size=1)
            t1 = time()
            PerfTrace.get_instance().event_complete(
                f"{self.next.__qualname__}.next.iter", MY_MODULE, t0, t1 - t0, arguments=profile_args)
            t0 = time()
            for batch in _dataset:
                t1 = time()
                PerfTrace.get_instance().event_complete(
                    f"{self.next.__qualname__}.next", MY_MODULE, t0, t1 - t0, arguments=profile_args)
                yield batch
                t0 = time()

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def finalize(self):
        pass
