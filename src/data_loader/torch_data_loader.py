from time import time
import logging
import math

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from src.common.constants import MODULE_DATA_LOADER
from src.common.enumerations import Shuffle, DatasetType
from src.data_loader.base_data_loader import BaseDataLoader
from src.reader.reader_factory import ReaderFactory
from src.utils.utility import utcnow, get_rank, Profile


class TorchDataset(Dataset):
    """
    Currently, we only support loading one sample per file
    TODO: support multiple samples per file
    """

    def __init__(self, format_type, dataset_type, epoch_number, num_samples, num_workers, samples_per_file):
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_DATA_LOADER, epoch=epoch_number):
            self.format_type = format_type
            self.dataset_type = dataset_type
            self.epoch_number = epoch_number
            self.num_samples = num_samples
            self.reader = None
            self.read_images = 0
            self.samples_per_file = samples_per_file
            if num_workers == 0:
                self.worker_init(-1)

    def worker_init(self, worker_id):
        with Profile(name=f"{self.worker_init.__qualname__}", cat=MODULE_DATA_LOADER, epoch=self.epoch_number):
            logging.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")
            self.reader = ReaderFactory.get_reader(type=self.format_type,
                                                   dataset_type=self.dataset_type,
                                                   thread_index=worker_id,
                                                   epoch_number=self.epoch_number)

    def __len__(self):
        with Profile(name=f"{self.__len__.__qualname__}", cat=MODULE_DATA_LOADER, epoch=self.epoch_number):
            return self.num_samples

    def __getitem__(self, idx):
        self.read_images += 1
        step = int(math.ceil(self.read_images/self.samples_per_file))
        logging.debug(f"{utcnow()} Rank {get_rank()} reading {idx} sample")
        with Profile(name=f"{self.__getitem__.__qualname__}", cat=MODULE_DATA_LOADER,
                     epoch=self.epoch_number,
                     step=step,
                     image_idx=idx):
            return self.reader.read_index(idx)


class TorchDataLoader(BaseDataLoader):

    def __init__(self, format_type, dataset_type, epoch_number):
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_DATA_LOADER, epoch=epoch_number):
            super().__init__(format_type, dataset_type, epoch_number)

    def read(self):
        with Profile(name=f"{self.read.__qualname__}", cat=MODULE_DATA_LOADER, epoch=self.epoch_number):
            do_shuffle = True if self._args.sample_shuffle != Shuffle.OFF    else False
            num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
            dataset = TorchDataset(self.format_type, self.dataset_type, self.epoch_number, num_samples, self._args.read_threads, self._args.num_samples_per_file)
            if do_shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            if self._args.read_threads > 1:
                prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
            else:
                prefetch_factor = self._args.prefetch_size
            if prefetch_factor > 0:
                if self._args.my_rank == 0:
                    logging.debug(
                        f"{utcnow()} Prefetch size is {self._args.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
            else:
                if self._args.my_rank == 0:
                    logging.debug(
                        f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
            logging.debug(f"{utcnow()} Setup dataloader with {self._args.read_threads} workers")
            batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
            self._dataset = DataLoader(dataset,
                                       batch_size=batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=True,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init,
                                       prefetch_factor=prefetch_factor if prefetch_factor > 0 else 2)  # 2 is the default value
            logging.debug(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataset) * batch_size} files")

        # self._dataset.sampler.set_epoch(epoch_number)

    def next(self):
        with Profile(name=f"{self.next.__qualname__}", cat=MODULE_DATA_LOADER, epoch=self.epoch_number):
            super().next()
            total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
            logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
            step = 1
            with Profile(name=f"{self.next.__qualname__}.next", cat=MODULE_DATA_LOADER) as lp:
                for batch in self._dataset:
                    lp.update(epoch=self.epoch_number, step=step).flush()
                    yield batch
                    step += 1
                    lp.reset()

    def finalize(self):
        with Profile(name=f"{self.next.__qualname__}", cat=MODULE_DATA_LOADER, epoch=self.epoch_number):
            pass
