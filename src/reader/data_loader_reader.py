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
        """
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return np.load(self.samples[idx])


    def __init__(self):
        super().__init__()
        self.read_threads = self._arg_parser.args.read_threads
        self.computation_threads = self._arg_parser.args.computation_threads

    def read(self, epoch_number, do_eval=False):
        # superclass function initializes the file list
        super().read(epoch_number, do_eval)

        do_shuffle = True if self.memory_shuffle != Shuffle.OFF else False
        
        if not do_eval:
            # Creates dataset objects using the file list
            train_dataset = self.NpzDataset(self._local_train_file_list)
            logging.debug("{} Rank {} train_dataset length is {}".format(utcnow(), self.my_rank, len(train_dataset)))

            # We can't use DistributedSampler as the splitting up of files between different ranks was already done
            # in the superclass method!
            train_sampler = DistributedSampler(train_dataset, 
                                                num_replicas=self.comm_size, 
                                                rank=self.my_rank,
                                                shuffle=do_shuffle, 
                                                seed=self.seed)
            logging.debug("{} Rank {} train_sampler length is {}".format(utcnow(), self.my_rank, len(train_sampler)))
            
            self._dataset_train = DataLoader(train_dataset,
                                        batch_size=self.batch_size,
                                        sampler=train_sampler,
                                        num_workers=self.read_threads,
                                        # Recommended to pin memory when using num_workers > 1 and GPUs 
                                        # See second warning https://pytorch.org/docs/stable/data.html#multi-process-data-loading
                                        pin_memory=True,
                                        drop_last=True)

            # https://discuss.pytorch.org/t/why-is-sampler-set-epoch-epoch-needed-for-distributedsampler/149672
            self._dataset_train.sampler.set_epoch(epoch_number)                    

            # TODO: If this becomes important, pytorch has a dif way of doing it.
            # We give it the number of batches to prefetch instead of a buffer size
            # See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
            # if self.prefetch:
            #     dataset_train = dataset_train.prefetch(buffer_size=self.prefetch_size)

        # We're evaluating, load the eval dataset
        else:
            val_dataset = self.NpzDataset(self._local_eval_file_list)

            # TODO: For image segmentation eval, they don't use a distributed sampler so we could parametrize this away if needed
            val_sampler = DistributedSampler(val_dataset, 
                                                num_replicas=self.comm_size, 
                                                rank=self.my_rank,
                                                shuffle=do_shuffle, 
                                                seed=self.seed)

            self._dataset_eval = DataLoader(val_dataset,
                                        batch_size=self.batch_size_eval,
                                        sampler=val_sampler,
                                        num_workers=self.read_threads,
                                        pin_memory=True,
                                        drop_last=False)

            self._dataset_eval.sampler.set_epoch(epoch_number)                    
        

    def next(self, do_eval=False):
        super().next()
        if do_eval:
            dataset = self._dataset_eval
        else:
            dataset = self._dataset_train 

        logging.debug("{} Rank {} should read {} batches".format(utcnow(), self.my_rank, len(dataset)))

        for batch in dataset:
            yield batch

    def finalize(self):
        pass
