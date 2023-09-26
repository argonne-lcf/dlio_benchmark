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
import math
import logging
from time import time

import nvidia.dali.fn as fn
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.dali_base_reader import DaliBaseReader
from dlio_benchmark.reader.tf_base_reader import TFBaseReader
from dlio_benchmark.utils.utility import utcnow, PerfTrace, Profile
from dlio_benchmark.common.enumerations import DatasetType, Shuffle
import nvidia.dali.tfrecord as tfrec

dlp = Profile(MODULE_DATA_READER)


class DaliNPZReader(DaliBaseReader):
    @dlp.log_init
    def __init__(self, dataset_type):
        super().__init__(dataset_type)

    @dlp.log
    def _load(self):
        logging.debug(
            f"{utcnow()} Reading {len(self.file_list)} files rank {self._args.my_rank}")
        random_shuffle = False
        seed = -1
        seed_change_epoch = False
        if self._args.sample_shuffle is not Shuffle.OFF:
            if self._args.sample_shuffle is not Shuffle.SEED:
                seed = self._args.seed
            random_shuffle = True
            seed_change_epoch = True
        initial_fill = 1024
        if self._args.shuffle_size > 0:
            initial_fill = self._args.shuffle_size
        prefetch_size = 1
        if self._args.prefetch_size > 0:
            prefetch_size = self._args.prefetch_size

        stick_to_shard = True
        if seed_change_epoch:
            stick_to_shard = False

        dataset = fn.readers.numpy(device='cpu', files=self.file_list, num_shards=self._args.comm_size,
                                   prefetch_queue_depth=prefetch_size, initial_fill=initial_fill,
                                   random_shuffle=random_shuffle, seed=seed, shuffle_after_epoch=seed_change_epoch,
                                   stick_to_shard=stick_to_shard, pad_last_batch=True)
        return dataset

    @dlp.log
    def finalize(self):
        pass
