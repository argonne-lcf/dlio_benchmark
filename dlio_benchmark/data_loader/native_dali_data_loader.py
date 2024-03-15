from time import time
import logging
import math
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DataLoaderType, DatasetType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow
from dlio_profiler.logger import dlio_logger as PerfTrace, fn_interceptor as Profile

dlp = Profile(MODULE_DATA_LOADER)


class NativeDaliDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.NATIVE_DALI)
        self.pipelines = []

    @dlp.log
    def read(self):
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        parallel = True if self._args.read_threads > 0 else False
        num_threads = 1
        if self._args.read_threads > 0:
            num_threads = self._args.read_threads
        # None executes pipeline on CPU and the reader does the batching
        logging.info(f"num_threads: {num_threads}; batch_size: {batch_size}")
        pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=None, 
                            py_num_workers=num_threads,
                            exec_async=False, exec_pipelined=False, 
                            py_start_method=self._args.multiprocessing_context)            
        with pipeline:
            images = ReaderFactory.get_reader(type=self.format_type,
                                            dataset_type=self.dataset_type,
                                            thread_index=-1,
                                            epoch_number=self.epoch_number).pipeline()
            pipeline.set_outputs(images)
        self.pipeline = pipeline
        self._dataset = DALIGenericIterator(self.pipeline, ['data'], auto_reset=True)
    @dlp.log
    def next(self):
        super().next()
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        for step in range(num_samples // batch_size):
            for batch in self._dataset:
                logging.debug(f"{utcnow()} Creating {len(batch)} batches by {self._args.my_rank} rank ")
                yield batch
        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)        
    @dlp.log
    def finalize(self):
        pass
