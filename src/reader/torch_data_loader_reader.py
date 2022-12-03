"""
   Copyright © 2022, UChicago Argonne, LLC
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
import numpy as np
from time import time
import os

from src.utils.utility import utcnow

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.common.enumerations import Shuffle, FormatType
from src.reader.reader_handler import FormatReader

from PIL import Image
import torchvision.transforms as transforms
import h5py

totensor=transforms.ToTensor()
### reading file of different formats. 
def read_jpeg(filename):
    return totensor(Image.open(filename))

def read_png(filename):
    return totensor(Image.open(filename))

def read_npz(filename):
    return np.load(filename)

def read_hdf5(f):
    file_h5 = h5py.File(f, 'r')
    d = file_h5['records'][:,:,:]
    l = file_h5['labels'][:]
    return d, l

def read_file(f):
    with open(f, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    return fileContent

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

    def __init__(self, dataset_type):
        super().__init__(dataset_type)
        self.read_threads = self._args.read_threads
        self.computation_threads = self._args.computation_threads
        self.format = self._args.format
        
    def read(self, epoch_number):
        # superclass function shuffle the file list 
        super().read(epoch_number)
        do_shuffle = True if self.memory_shuffle != Shuffle.OFF else False
        
        dataset = TorchDataset(self._file_list, self.my_rank, self.format)
        
        # TODO: In image segmentation, the distributed sampler is not used during eval, we could parametrize this away if needed
        # This handles the partitioning between ranks
        sampler = DistributedSampler(dataset, 
                                     num_replicas=self.comm_size, 
                                     rank=self.my_rank,
                                     shuffle=do_shuffle, 
                                     seed=self.seed)
        logging.debug(f"{utcnow()} Rank {self.my_rank} length of distributed sampler {len(sampler)} ")
        
        if self.read_threads>1:
            prefetch_factor = math.ceil(self.prefetch_size / self.read_threads)
        else:
            prefetch_factor = self.prefetch_size
        if prefetch_factor>0:
            if self.my_rank==0:
                logging.info(f"{utcnow()} Prefetch size is {prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            if self.my_rank==0:
                logging.info(f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
        self._dataset = DataLoader(dataset,
                                   batch_size=self.batch_size,
                                   sampler=sampler,
                                   num_workers=self.read_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   prefetch_factor=prefetch_factor if prefetch_factor >0 else 2) # 2 is the default value


        logging.debug(f"{utcnow()} Rank {self.my_rank} will read {len(self._dataset) * self.batch_size} files")

        self._dataset.sampler.set_epoch(epoch_number)
        # Must set the epoch in DistributedSampler to ensure proper shuffling
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

    def next(self):
        super().next()
        logging.debug(f"{utcnow()} Rank {self.my_rank} should read {len(self._dataset)} batches")

        for batch in self._dataset:   
            yield batch

    def finalize(self):
        pass
