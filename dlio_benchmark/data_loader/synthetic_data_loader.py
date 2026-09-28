"""
   Copyright (c) 2025, UChicago Argonne, LLC
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
import time
import torch

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.utils.utility import utcnow, Profile, dft_ai

dlp = Profile(MODULE_DATA_LOADER)


class SyntheticDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.SYNTHETIC)
        self.iter_time = self._args.iter_time
        shape = self._args.resized_image.shape
        # Calculate local samples for this rank
        total_samples = self.num_samples
        samples_per_proc = int(math.ceil(total_samples / self._args.comm_size))
        start_sample = self._args.my_rank * samples_per_proc
        end_sample = (self._args.my_rank + 1) * samples_per_proc - 1
        if end_sample > total_samples - 1:
            end_sample = total_samples - 1
        local_num_samples = end_sample - start_sample + 1

        # Calculate number of batches
        self.num_batches = int(math.ceil(local_num_samples / self.batch_size))

        # Pre-create a batch matching what the pytorch DataLoader would produce:
        # each sample is resized_image with shape (H, W), collated into (B, H, W).
        # Use resized_image.shape directly so this works for any dimensionality.
        self.zero_batch = torch.zeros(
            (self.batch_size, *shape),
            dtype=torch.uint8,
        )
        if torch.cuda.is_available():
            self.zero_batch = self.zero_batch.pin_memory()

    @dlp.log
    def read(self, init=False):
        return

    @dft_ai.data.item
    def getitem(self):
        return self.zero_batch

    @dlp.log
    def next(self):
        super().next()
        self.logger.debug(
            f"{utcnow()} Iterating pipelines by {self._args.my_rank} rank "
        )
        self.read(True)

        step = 1
        dft_ai.dataloader.fetch.start()
        while step <= self.num_batches:
            dft_ai.dataloader.fetch.stop()
            dft_ai.update(step=step)
            step += 1
            time.sleep(self.iter_time)
            yield self.getitem()
            dft_ai.dataloader.fetch.start()

        self.epoch_number += 1
        dft_ai.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        return
