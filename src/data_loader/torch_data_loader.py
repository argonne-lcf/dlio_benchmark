from time import time
import logging
import math

from torch.utils.data import Dataset, DistributedSampler, DataLoader, RandomSampler, SequentialSampler

from src.common.enumerations import Shuffle, DataLoaderType, DatasetType
from src.data_loader.base_data_loader import BaseDataLoader
from src.reader.reader_factory import ReaderFactory
from src.utils.utility import utcnow, get_rank, timeit, PerfTrace,event_logging

MY_MODULE="data_loader"
profile_args = {}
class TorchDataset(Dataset):
    """
    Currently, we only support loading one sample per file
    TODO: support multiple samples per file
    """

    def __init__(self, format_type, dataset_type, epoch_number, num_samples, num_workers):
        global profile_args
        profile_args["epoch"] = epoch_number
        t0 = time()
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch_number
        self.num_samples = num_samples
        self.reader = None
        if num_workers == 0:
            self.worker_init(-1)
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0, arguments=profile_args)

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def worker_init(self, worker_id):
        logging.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=worker_id,
                                               epoch_number=self.epoch_number)

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        global profile_args
        profile_args["image_idx"] = idx
        t0 = time()
        logging.debug(f"{utcnow()} Rank {get_rank()} reading {idx} sample")
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__getitem__.__qualname__}", MY_MODULE, t0, t1 - t0,
                                                arguments=profile_args)
        return self.reader.read_index(idx)


class TorchDataLoader(BaseDataLoader):

    def __init__(self, format_type, dataset_type, epoch_number):
        global profile_args
        profile_args["epoch"] = epoch_number
        t0 = time()
        super().__init__(format_type, dataset_type, epoch_number)
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0, arguments=profile_args)

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def read(self):
        do_shuffle = True if self._args.sample_shuffle != Shuffle.OFF else False
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        dataset = TorchDataset(self.format_type, self.dataset_type, self.epoch_number, num_samples, self._args.read_threads)
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

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def next(self):
        global profile_args
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        count = 1
        t0 = time()
        for batch in self._dataset:
            profile_args["step"] = count
            t1 = time()
            PerfTrace.get_instance().event_complete(
                f"{self.next.__qualname__}.next", MY_MODULE, t0, t1 - t0, arguments=profile_args)
            yield batch
            count += 1
            t0 = time()

    @event_logging(module=MY_MODULE, arguments=profile_args)
    def finalize(self):
        pass
