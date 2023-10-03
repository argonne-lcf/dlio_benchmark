from time import time
import logging
import math
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DataLoaderType, DatasetType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, get_rank, timeit, Profile

dlp = Profile(MODULE_DATA_LOADER)


class DaliDataset(object):

    def __init__(self, format_type, dataset_type, epoch, num_samples, batch_size, thread_index):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch = epoch
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.worker_index = thread_index
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=thread_index,
                                               epoch_number=self.epoch)

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch * self.num_samples + self.worker_index
        step = int(math.ceil(sample_idx / self.batch_size))
        logging.debug(f"{utcnow()} Reading {sample_idx} {sample_info.iteration} {self.num_samples} {self.worker_index}")
        if sample_info.iteration >= self.num_samples:
            # Indicate end of the epoch
            raise StopIteration()
        with Profile(MODULE_DATA_LOADER, epoch=self.epoch, image_idx=sample_idx, step=step):
            image = self.reader.read_index(sample_idx, step)
        return image, np.uint8([sample_idx])


class DaliDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.DALI)
        self.pipeline = None

    @dlp.log
    def read(self):
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        parallel = True if self._args.read_threads > 0 else False
        self.pipelines = []
        num_threads = 1
        if self._args.read_threads > 0:
            num_threads = self._args.read_threads
        prefetch_size = 2
        if self._args.prefetch_size > 0:
            prefetch_size = self._args.prefetch_size
        samples_per_worker = num_samples // batch_size // num_threads
        for worker_index in range(num_threads):
            # None executes pipeline on CPU and the reader does the batching
            dataset = DaliDataset(self.format_type, self.dataset_type, self.epoch_number, samples_per_worker,
                                  batch_size, worker_index)
            pipeline = Pipeline(batch_size=batch_size, num_threads=1, device_id=None, py_num_workers=1,
                                prefetch_queue_depth=prefetch_size, py_start_method='fork', exec_async=True)
            with pipeline:
                images, labels = fn.external_source(source=dataset, num_outputs=2, dtype=[types.UINT8, types.UINT8],
                                                    parallel=True, batch=False)
                pipeline.set_outputs(images, labels)
            self.pipelines.append(pipeline)
        for pipe in self.pipelines:
            pipe.start_py_workers()
        for pipe in self.pipelines:
            pipe.build()
        for pipe in self.pipelines:
            pipe.schedule_run()
        logging.debug(f"{utcnow()} Starting {num_threads} pipelines by {self._args.my_rank} rank ")


    @dlp.log
    def next(self):
        super().next()
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        # DALIGenericIterator(self.pipelines, ['data', 'label'])

        logging.debug(f"{utcnow()} Iterating pipelines by {self._args.my_rank} rank ")
        step = 0
        while step <= num_samples // batch_size:
            for pipe in self.pipelines:
                outputs = pipe.share_outputs()
                logging.debug(f"{utcnow()} Output batch {step} {len(outputs)}")
                for batch in outputs:
                    yield batch
                    step += 1
                pipe.release_outputs()
                pipe.schedule_run()
                
                
    @dlp.log
    def finalize(self):
        pass
