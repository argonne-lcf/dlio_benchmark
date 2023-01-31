import math
import logging
import numpy as np

from src.utils.utility import utcnow

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from src.common.enumerations import Shuffle
from src.reader.reader_handler import FormatReader
import os
import torch

import math
import logging
import numpy as np

from src.utils.utility import utcnow

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from src.common.enumerations import Shuffle, DatasetType
from src.reader.reader_handler import FormatReader
import os
import torch

class TeraBinLoaderReader(FormatReader):
    """
    Terabyte DataLoader reader and iterator logic.
    PyTorch data loader is file format agnostic so it is not technically separate from the other formats
    This is a revised version data_loader_reader to reflect the data loader for binary terabyte dataset for DLRM
    """

    class TerabyteBinDataset(Dataset):
        def __init__(self, data_file, batch_size=1, max_ind_range=-1, bytes_per_feature=4):
            self.tar_fea = 1   # single target
            self.den_fea = 13  # 13 dense  features
            self.spa_fea = 26  # 26 sparse features
            self.tad_fea = self.tar_fea + self.den_fea
            self.tot_fea = self.tad_fea + self.spa_fea
            self.batch_size = batch_size
            self.max_ind_range = max_ind_range
            self.bytes_per_entry = (bytes_per_feature * self.tot_fea * batch_size)

            self.num_entries = math.ceil(os.path.getsize(data_file) / self.bytes_per_entry)

            print('data file:', data_file, 'number of batches:', self.num_entries)
            self.file = open(data_file, 'rb')

        def __len__(self):
            return self.num_entries
        
        def __getitem__(self, idx):
            self.file.seek(idx * self.bytes_per_entry, 0)
            raw_data = self.file.read(self.bytes_per_entry)
            array = np.frombuffer(raw_data, dtype=np.int32)
            tensor = torch.from_numpy(array).view((-1, self.tot_fea))

            return _transform_features(x_int_batch=tensor[:, 1:14],
                                    x_cat_batch=tensor[:, 14:],
                                    y_batch=tensor[:, 0],
                                    max_ind_range=self.max_ind_range,
                                    flag_input_torch_tensor=True)

        def __del__(self):
            self.file.close()

    def __init__(self, dataset_type):
        super().__init__(dataset_type)
        self.read_threads = self._arg_parser.args.read_threads
        self.computation_threads = self._arg_parser.args.computation_threads

    def read(self, epoch_number):
        # superclass function initializes the file list
        super().read(epoch_number)

        do_shuffle = True if self.memory_shuffle != Shuffle.OFF else False

        # There's only one training and valid file with shared file access.
        if self.dataset_type == DatasetType.TRAIN:
            dataset = self.TerabyteBinDataset(self._file_list[0], batch_size=self.batch_size, max_ind_range=10000000)

            self._dataset = DataLoader(
                dataset,
                batch_size=None,
                batch_sampler=None,
                shuffle=do_shuffle,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
                sampler=RandomSampler(dataset)
            ) 
        elif self.dataset_type == DatasetType.VALID:
            dataset = self.TerabyteBinDataset(self._file_list[0], batch_size=self.batch_size, max_ind_range=10000000)

            self._dataset = DataLoader(
                dataset,
                batch_size=None,
                batch_sampler=None,
                shuffle=do_shuffle,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
            ) 

        # Must set the epoch in DistributedSampler to ensure proper shuffling
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        # self._dataset.sampler.set_epoch(epoch_number)

        logging.debug(f"{utcnow()} Rank {self.my_rank} will read {len(self._dataset) * self.batch_size} files")

    def next(self):
        super().next()

        dataset = self._dataset
        logging.debug(f"{utcnow()} Rank {self.my_rank} should read {len(dataset)} batches")

        for batch in dataset:
            yield batch

    def finalize(self):
        pass


def _transform_features(
    x_int_batch, x_cat_batch, y_batch, max_ind_range, flag_input_torch_tensor=False
):
    if max_ind_range > 0:
        x_cat_batch = x_cat_batch % max_ind_range

    if flag_input_torch_tensor:
        x_int_batch = torch.log(x_int_batch.clone().detach().type(torch.float) + 1)
        x_cat_batch = x_cat_batch.clone().detach().type(torch.long)
        y_batch = y_batch.clone().detach().type(torch.float32).view(-1, 1)
    else:
        x_int_batch = torch.log(torch.tensor(x_int_batch, dtype=torch.float) + 1)
        x_cat_batch = torch.tensor(x_cat_batch, dtype=torch.long)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

    batch_size = x_cat_batch.shape[0]
    feature_count = x_cat_batch.shape[1]
    lS_o = torch.arange(batch_size).reshape(1, -1).repeat(feature_count, 1)

    return x_int_batch, lS_o, x_cat_batch.t(), y_batch.view(-1, 1)