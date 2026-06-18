"""
   Copyright (c) 2026, Enakta Labs, LTD
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
import os
import io
import numpy as np
import math
import pickle
from torch.utils.data import Dataset

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType, FormatType
from dlio_benchmark.data_loader.torch_data_loader import BaseTorchDataLoader
from dlio_benchmark.utils.utility import utcnow, DLIOMPI, Profile, dft_ai
from dlio_benchmark.utils.config import ConfigArguments

dlp = Profile(MODULE_DATA_LOADER)


def get_format_reader(format):
    if format == FormatType.NPZ:
        return lambda b: np.load(io.BytesIO(b), allow_pickle=True)["x"]
    elif format == FormatType.NPY:
        return lambda b: np.load(io.BytesIO(b), allow_pickle=True)
    else:
        raise ValueError(f"TorchDaosDataset does not support {format} format")

class TorchDaosDataset(Dataset):
    """
    Wrapper over DaosDataset to log calls for the profiler
    """
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.num_samples = num_samples
        self.num_images_read = 0
        self.batch_size = batch_size
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        self.logger = args.logger
        self.dlp_logger = None

        # to avoid loading pydoas.torch at the top level if not needed or not installed
        from pydaos.torch import Dataset as DaosDataset

        prefix = os.path.join(args.data_folder, f"{self.dataset_type}")
        self.dataset = DaosDataset(pool=args.daos_pool,
                                cont=args.daos_cont,
                                path=prefix,
                                transform_fn=get_format_reader(self.format_type))

        # self.num_samples = len(self.dataset)
        if num_workers == 0:
            self.worker_init(-1)


    @dlp.log
    def worker_init(self, worker_id):
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        self.dlp_logger = _args.configure_dftracer(is_child=True, use_pid=True)
        self.logger.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")
        self.dataset.worker_init(worker_id)

    def __del__(self):
        if self.dlp_logger:
            self.dlp_logger.finalize()

    @dlp.log
    def __len__(self):
        return self.num_samples

    @dlp.log
    def __getitem__(self, image_idx):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        self.logger.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {image_idx} sample")
        dlp.update(step=step)
        dft_ai.update(step=step)
        return self.dataset.__getitem__(image_idx)

    @dlp.log
    def __getitems__(self, indices):
        self.num_images_read += len(indices)
        step = int(math.ceil(self.num_images_read / self.batch_size))
        self.logger.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {len(indices)} samples")
        dlp.update(step=step)
        dft_ai.update(step=step)
        return self.dataset.__getitems__(indices)

class DaosTorchDataLoader(BaseTorchDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.DAOS_PYTORCH)

    def get_dataset(self) -> Dataset:
        return TorchDaosDataset(self.format_type, self.dataset_type, self.epoch_number, self.num_samples,
                                self.read_threads, self.batch_size)
