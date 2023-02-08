import logging
import math

from torch.utils.data import Dataset, DistributedSampler, DataLoader

from src.common.enumerations import Shuffle, DataLoaderType
from src.data_loader.base_data_loader import BaseDataLoader
from src.reader.reader_factory import ReaderFactory
from src.utils.utility import utcnow, get_rank


class TorchDataset(Dataset):
    """
    Currently, we only support loading one sample per file
    TODO: support multiple samples per file
    """

    def __init__(self, format_type, dataset_type, epoch_number):
        self.reader = ReaderFactory.get_reader(format_type,
                                               dataset_type=dataset_type)
        self.reader.read(epoch_number)

    def __len__(self):
        return self.reader.get_sample_len()

    #@timeit
    def __getitem__(self, idx):
        logging.debug(f"{utcnow()} Rank {get_rank()} reading {idx} sample")
        return self.reader.read_index(idx)


class TorchDataLoader(BaseDataLoader):

    def __init__(self, format_type, dataset_type):
        super().__init__(format_type, dataset_type)
        self.read_threads = self._args.read_threads
        self.computation_threads = self._args.computation_threads
        self.format = self._args.format

    def read(self, epoch_number):
        do_shuffle = True if self.sample_shuffle != Shuffle.OFF else False
        dataset = TorchDataset(self.format, self.dataset_type, epoch_number)
        # TODO: In image segmentation, the distributed sampler is not used during eval, we could parametrize this away if needed
        # This handles the partitioning between ranks
        sampler = DistributedSampler(dataset,
                                     num_replicas=self.comm_size,
                                     rank=self.my_rank,
                                     shuffle=do_shuffle,
                                     seed=self.seed)
        logging.debug(f"{utcnow()} Rank {self.my_rank} length of distributed sampler {len(sampler)} ")

        if self.read_threads > 1:
            prefetch_factor = math.ceil(self.prefetch_size / self.read_threads)
        else:
            prefetch_factor = self.prefetch_size
        if prefetch_factor > 0:
            if self.my_rank == 0:
                logging.info(
                    f"{utcnow()} Prefetch size is {self.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            if self.my_rank == 0:
                logging.info(
                    f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
        self._dataset = DataLoader(dataset,
                                   batch_size=self.batch_size,
                                   sampler=sampler,
                                   num_workers=self.read_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   prefetch_factor=prefetch_factor if prefetch_factor > 0 else 2)  # 2 is the default value
        logging.debug(f"{utcnow()} Rank {self.my_rank} will read {len(self._dataset) * self.batch_size} files")

        self._dataset.sampler.set_epoch(epoch_number)

    def next(self):
        super().next()
        logging.debug(f"{utcnow()} Rank {self.my_rank} should read {len(self._dataset)} batches")
        for batch in self._dataset:
            yield batch

    def finalize(self):
        pass