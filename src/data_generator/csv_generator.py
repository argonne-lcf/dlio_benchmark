from src.data_generator.data_generator import DataGenerator
import math
import os

from numpy import random
import csv

from shutil import copyfile
from src.utils.utility import progress


class CSVGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        super().generate()
        record = random.random((self._dimension * self._dimension))
        record_label = 0
        prev_out_spec = ""
        count = 0
        for i in range(0, int(self.num_files)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.num_files, "Generating CSV Data")
                out_path_spec = "{}_{}_of_{}.csv".format(self._file_prefix, i, self.num_files)
                if count == 0:
                    prev_out_spec = out_path_spec
                    with open(out_path_spec, 'w') as csvfile:
                        writer = csv.writer(csvfile)
                        print("{} samples of size {}".format(self.num_samples, self._dimension * self._dimension))
                        for j in range(0, self.num_samples):
                            writer.writerow([record, record_label])
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)