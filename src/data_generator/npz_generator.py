from src.data_generator.data_generator import DataGenerator

import numpy as np
from numpy import random

from src.utils.utility import progress
from shutil import copyfile


class NPZGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        super().generate()
        records = random.random((self._dimension, self._dimension, self.num_samples))
        record_labels = [0] * self.num_samples
        prev_out_spec =""
        count = 0
        for i in range(0, int(self.num_files)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.num_files, "Generating NPZ Data")
                out_path_spec = "{}_{}_of_{}.csv".format(self._file_prefix, i, self.num_files)
                if count == 0:
                    prev_out_spec = out_path_spec
                    np.savez(out_path_spec, x=records, y=record_labels)
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)