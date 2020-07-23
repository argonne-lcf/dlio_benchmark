from src.data_generator.data_generator import DataGenerator
import math
import os

from numpy import random
import csv

class CSVGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        super().generate()
        record = random.random((self._dimension * self._dimension))
        record_label = 0
        for i in range(0, int(self.num_files)):
            out_path_spec = "{}_{}_of_{}.csv".format(self._file_prefix, i, self.num_files)
            with open(out_path_spec, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for j in range(0, self.num_samples):
                    writer.writerow([record, record_label])