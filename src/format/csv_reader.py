from src.common.enumerations import Shuffle
from src.format.reader_handler import FormatReader
import csv
import math

from numpy import random


class CSVReader(FormatReader):
    def __init__(self):
        super().__init__()

    def read(self, epoch_number):
        super().read(epoch_number)
        packed_array = []
        for file in self._local_file_list:
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file)
                rows = []
                for row in csv_reader:
                    rows.append({
                        'record': row[0],
                        'label': row[1]
                    })
                packed_array.append({
                    'dataset': rows,
                    'current_sample': 0,
                    'total_samples': len(rows)
                })
        self._dataset = packed_array

    def next(self):
        super().next()
        for element in self._dataset:
            current_index = element['current_sample']
            total_samples = element['total_samples']
            num_sets = list(range(0, int(math.ceil(total_samples / self.batch_size))))
            if self.memory_shuffle != Shuffle.OFF:
                if self.memory_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(num_sets)
            for num_set in num_sets:
                yield element['dataset'][num_set * self.batch_size:(num_set + 1) * self.batch_size - 1]

    def finalize(self):
        pass
