"""
   Copyright 2021 UChicago Argonne, LLC

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
import numpy as np
from time import time

from src.utils.utility import utcnow

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.common.enumerations import Shuffle, FormatType
from src.reader.reader_handler import FormatReader

from torchvision.io import read_image, decode_png, decode_jpeg, read_file

### reading file of different formats. 
def read_jpeg(filename):
    return read_image(filename)

def read_png(filename):
    return read_image(filename)

def read_npz(filename):
    return np.load(filename)

def read_hdf5(f):
    file_h5 = h5py.File(file, 'r')
    d = file_h5['x']
    return d

filereader={
    FormatType.JPEG: read_jpeg, 
    FormatType.PNG: read_png, 
    FormatType.NPZ: read_npz, 
    FormatType.HDF5: read_hdf5, 
}

class TorchDataset(Dataset):
        """
        Currently, we only support loading one sample per file 
        TODO: support multiple samples per file
        """
        def __init__(self, samples, rank, format):
            self.samples = samples
            self.my_rank = rank
            try:
                self.read = filereader[format]
            except:
                print(f"Unsupported file reader for {format}. Reading the byte contents instead")
                self.read = read_file
            
        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            logging.debug(f"{utcnow()} Rank {self.my_rank} reading {self.samples[idx]}")
            return self.read(self.samples[idx])

class TorchDataLoaderReader(FormatReader):
    """
    DataLoader reader and iterator logic.
    """

    def __init__(self):
        super().__init__()
        self.read_threads = self._args.read_threads
        self.computation_threads = self._args.computation_threads
        self.format = self._args.format

    def read(self, epoch_number, do_eval=False):
        # superclass function initializes the file list
        super().read(epoch_number, do_eval)

        do_shuffle = True if self.memory_shuffle != Shuffle.OFF else False

        dataset = TorchDataset(self._local_file_list, self.my_rank, self.format)

        # TODO: In image segmentation, the distributed sampler is not used during eval, we could parametrize this away if needed
        # This handles the partitioning between ranks
        sampler = DistributedSampler(dataset, 
                                num_replicas=self.comm_size, 
                                rank=self.my_rank,
                                shuffle=do_shuffle, 
                                seed=self.seed)

        self._dataset = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    sampler=sampler,
                                    num_workers=self.read_threads,
                                    pin_memory=True,
                                    drop_last=True,
                                    prefetch_factor=self.prefetch_size if self.prefetch_size > 0 else 2) # 2 is the default value

        # Must set the epoch in DistributedSampler to ensure proper shuffling
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        self._dataset.sampler.set_epoch(epoch_number)

        logging.debug(f"{utcnow()} Rank {self.my_rank} will read {len(self._dataset) * self.batch_size} files")

    def next(self):
        super().next()

        dataset = self._dataset
        logging.debug(f"{utcnow()} Rank {self.my_rank} should read {len(dataset)} batches")

        for batch in dataset:   
            yield batch

    def finalize(self):
        pass
