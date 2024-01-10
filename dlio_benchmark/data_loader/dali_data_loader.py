"""
   Copyright (c) 2022, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from time import time
import logging
import math
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DataLoaderType, DatasetType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow
from dlio_profiler.logger import fn_interceptor as Profile

dlp = Profile(MODULE_DATA_LOADER)


class DaliDataset(object):

    def __init__(self, format_type, dataset_type, epoch, thread_index,
                 total_num_workers, total_num_samples, samples_per_worker, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch = epoch
        self.total_num_workers = total_num_workers
        self.total_num_samples = total_num_samples
        self.samples_per_worker = samples_per_worker
        self.batch_size = batch_size
        self.worker_index = thread_index
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=thread_index,
                                               epoch_number=self.epoch)

    def __call__(self, sample_info):
        logging.debug(
            f"{utcnow()} Reading {sample_info.idx_in_epoch} out of {self.samples_per_worker} by worker {self.worker_index}")
        sample_idx = sample_info.idx_in_epoch + self.total_num_workers * self.worker_index
        logging.debug(
            f"{utcnow()} Reading {sample_idx} on {sample_info.iteration} by worker {self.worker_index}")
        if sample_info.iteration >= self.samples_per_worker or sample_idx >= self.total_num_samples:
            # Indicate end of the epoch
            raise StopIteration()

        step = int(math.ceil(sample_idx / self.batch_size))
        with Profile(MODULE_DATA_LOADER, epoch=self.epoch, image_idx=sample_idx, step=step):
            image = self.reader.read_index(sample_idx, step)
        return image, np.uint8([sample_idx])

class DaliDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.DALI)
        self.pipelines = []

    @dlp.log
    def read(self):
        parallel = True if self._args.read_threads > 0 else False
        self.pipelines = []
        num_threads = 1
        if self._args.read_threads > 0:
            num_threads = self._args.read_threads
        prefetch_size = 2
        if self._args.prefetch_size > 0:
            prefetch_size = self._args.prefetch_size
        num_pipelines = 1
        samples_per_worker = self.num_samples // num_pipelines // self._args.comm_size

        for worker_index in range(num_pipelines):
            global_worker_index = self._args.my_rank * num_pipelines + worker_index
            # None executes pipeline on CPU and the reader does the batching
            dataset = DaliDataset(self.format_type, self.dataset_type, self.epoch_number, global_worker_index,
                                  self._args.comm_size * num_pipelines, self.num_samples, samples_per_worker, self.batch_size)
            pipeline = Pipeline(batch_size=self.batch_size, num_threads=num_threads, device_id=None, py_num_workers=num_threads//num_pipelines,
                                prefetch_queue_depth=prefetch_size, py_start_method=self._args.multiprocessing_context, exec_async=True)
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
        # DALIGenericIterator(self.pipelines, ['data', 'label'])

        logging.debug(f"{utcnow()} Iterating pipelines by {self._args.my_rank} rank ")
        step = 0
        while step <= self.num_samples // self.batch_size:
            for pipe in self.pipelines:
                outputs = pipe.share_outputs()
                logging.debug(f"{utcnow()} Output batch {step} {len(outputs)}")
                yield outputs
                step += 1
                pipe.release_outputs()
                pipe.schedule_run()
                
                
    @dlp.log
    def finalize(self):
        pass
