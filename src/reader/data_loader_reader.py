"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of DLProfile
 DLIO is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
"""
import math
import logging
import numpy as np

from src.utils.utility import utcnow

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.common.enumerations import Shuffle
from src.reader.reader_handler import FormatReader

class DataLoaderReader(FormatReader):
    """
    DataLoader reader and iterator logic.
    PyTorch data loader is file format agnostic so it is not technically separate from the other formats
    For now we will assume we are reading .npz files through the dataloader
    """
    class NpzDataset(Dataset):
        """
        Only support a single list of files for now 
        Could be extended to support different X and Y files, like in imseg
        TODO: Move the Dataset somewhere else and support other file formats.
        """
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            logging.debug(f"{utcnow()} Reading {self.samples[idx]}")
            return np.load(self.samples[idx])


    def __init__(self):
        super().__init__()
        self.read_threads = self._arg_parser.args.read_threads
        self.computation_threads = self._arg_parser.args.computation_threads

    def read(self, epoch_number, do_eval=False):
        # superclass function initializes the file list
        super().read(epoch_number, do_eval)

        do_shuffle = True if self.memory_shuffle != Shuffle.OFF else False

        dataset = self.NpzDataset(self._local_file_list)

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
