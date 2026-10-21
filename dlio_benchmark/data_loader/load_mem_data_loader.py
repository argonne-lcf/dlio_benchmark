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
import numpy as np
import time
import torch

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.data_loader.torch_data_loader import TorchDataLoader
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_LOADER)
class LoadMemDataLoader(BaseDataLoader):
    """
    Load Memory Data Loader - Preloads entire dataset into memory for fast access.

    Uses PyTorch's DataLoader internally to read all data once, stores it in
    memory, and then serves batches from the preloaded cache.  Batches are
    stored exactly as the pytorch DataLoader produces them (no channel
    expansion or reshaping) so that the data flowing into the model is
    identical to the pytorch loader path.
    """

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.LOAD_MEM)
        self.logger.info(f"reader.iter_time={self._args.iter_time}")
        self._torch_loader = TorchDataLoader(format_type, dataset_type, epoch)
        self._preloaded_batches = []

    @dlp.log
    def read(self, init=False):
        """Preload all data into memory using the PyTorch dataloader"""
        if self._preloaded_batches:
            # Already preloaded, no need to read again
            return

        self._torch_loader.read()

        self.logger.info("Preloading data")
        batch_num = 0
        for batch in self._torch_loader.next():
            # Pin the batch for faster H2D transfer, but do NOT reshape.
            # The batch shape must stay identical to what the pytorch
            # DataLoader yields so that all three loader modes feed the
            # same tensor shape into the model's validate_data.
            if torch.cuda.is_available() and isinstance(batch, torch.Tensor) and not batch.is_pinned():
                batch = batch.pin_memory()
            elif isinstance(batch, (tuple, list)):
                pinned = []
                for t in batch:
                    if torch.cuda.is_available() and isinstance(t, torch.Tensor) and not t.is_pinned():
                        t = t.pin_memory()
                    pinned.append(t)
                batch = type(batch)(pinned)

            self._preloaded_batches.append(batch)
            batch_num += 1

        if self._args.my_rank == 0:
            self.logger.info(f"Preloaded {len(self._preloaded_batches)} batches into memory")

    @dlp.log
    def next(self):
        """Yield batches from preloaded data"""
        super().next()
        if not self._preloaded_batches:
            self.read(True)

        if not self._preloaded_batches:
            self.logger.warning("No preloaded batches are available for iteration.")
            return

        # Yield from preloaded batches
        step = 1
        for batch in dlp.iter(self._preloaded_batches):
            dlp.update(step=step)
            step += 1
            time.sleep(self._args.iter_time)
            yield batch

    @dlp.log
    def finalize(self):
        """Clean up resources"""
        self._torch_loader.finalize()
        # Keep _preloaded_batches in memory across epochs
        return
