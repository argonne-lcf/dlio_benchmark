from src.common.enumerations import Shuffle, FileAccess
from src.reader.reader_handler import FormatReader
import numpy as np
import math
from numpy import random

from src.utils.utility import progress


class NPZReader(FormatReader):
    def __init__(self):
        super().__init__()

    def read(self, epoch_number):
        super().read(epoch_number)
        packed_array = []
        for file in self._local_file_list:
            with np.load(file, allow_pickle=True) as data:
                rows = data['x']
                packed_array.append({
                    'dataset': rows,
                    'current_sample': 0,
                    'total_samples': rows.shape[2]
                })
        self._dataset =  packed_array

    def next(self):
        super().next()
        total = 0
        count = 1
        for element in self._dataset:
            current_index = element['current_sample']
            total_samples = element['total_samples']
            if FileAccess.MULTI == self.file_access:
                num_sets = list(range(0, int(math.ceil(total_samples / self.batch_size))))
            else:
                total_samples_per_rank = int(total_samples / self.comm_size)
                part_start, part_end = (int(total_samples_per_rank * self.my_rank / self.batch_size),
                                        int(total_samples_per_rank * (self.my_rank + 1) / self.batch_size))
                num_sets = list(range(part_start, part_end))
            total += len(num_sets)
            if self.memory_shuffle != Shuffle.OFF:
                if self.memory_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(num_sets)
            for num_set in num_sets:
                progress(count, total, "Reading NPZ Data")
                count += 1
                yield element['dataset'][:][:][num_set * self.batch_size:(num_set + 1) * self.batch_size - 1]

    def finalize(self):
        pass