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
import os.path

import math
import logging
from time import time, sleep
import numpy as np

import nvidia
import nvidia.dali.fn as fn
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import utcnow
from dlio_benchmark.common.enumerations import DatasetType, Shuffle
import nvidia.dali.tfrecord as tfrec
from dlio_profiler.logger import dlio_logger as PerfTrace, fn_interceptor as Profile

dlp = Profile(MODULE_DATA_READER)


class DaliTFRecordReader(FormatReader):
    """
    Reader for NPZ files
    """    
    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

    @dlp.log
    def open(self, filename):
        super().open(filename)
    
    def close(self):
        super().close()

    @dlp.log
    def pipeline(self):
        folder = "valid"
        if self.dataset_type == DatasetType.TRAIN:
            folder = "train"
        index_folder = f"{self._args.data_folder}/index/{folder}"
        index_files = []
        for file in self._file_list:
            filename = os.path.basename(file)
            index_files.append(f"{index_folder}/{filename}.idx")
        logging.info(
            f"{utcnow()} Reading {len(self.file_list)} files rank {self._args.my_rank}")
        random_shuffle = False
        seed = -1
        if self._args.sample_shuffle is not Shuffle.OFF:
            if self._args.sample_shuffle is not Shuffle.SEED:
                seed = self._args.seed
            random_shuffle = True
        initial_fill = 1024
        if self._args.shuffle_size > 0:
            initial_fill = self._args.shuffle_size
        prefetch_size = 1
        if self._args.prefetch_size > 0:
            prefetch_size = self._args.prefetch_size
        dataset = fn.readers.tfrecord(path=self._file_list,
                                      index_path=index_files,
                                      features={
                                          'image': tfrec.FixedLenFeature((), tfrec.string, ""),
                                          'size': tfrec.FixedLenFeature([1], tfrec.int64, 0)
                                      }, num_shards=self._args.comm_size,
                                      prefetch_queue_depth=prefetch_size,
                                      initial_fill=initial_fill,
                                      random_shuffle=random_shuffle, seed=seed,
                                      stick_to_shard=True, pad_last_batch=True, 
                                      dont_use_mmap=self._args.dont_use_mmap)
        dataset = self._resize(dataset['image'])
        fn.python_function(dataset, function=self.preprocess, num_outputs=0)
        return dataset

    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        raise Exception("get sample method is not implemented in dali readers")

    def next(self):
        super().next()
        raise Exception("next method is not implemented in dali readers")

    def read_index(self):
        super().read_index()
        raise Exception("read_index method is not implemented in dali readers")

    @dlp.log
    def _resize(self, dataset):
        return fn.resize(dataset, size=[self._args.max_dimension, self._args.max_dimension])

    @dlp.log
    def finalize(self):
        pass
