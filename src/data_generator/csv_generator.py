from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator
import math
import os

from numpy import random
import csv

from shutil import copyfile
from src.utils.utility import progress
import pandas as pd


class CSVGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        super().generate()
        record = random.random((self._dimension * self._dimension))
        records = [record]*self.num_samples
        record_label = 0
        prev_out_spec = ""
        count = 0
        for i in range(0, int(self.num_files)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.num_files, "Generating CSV Data")
                out_path_spec = "{}_{}_of_{}.csv".format(self._file_prefix, i, self.num_files)
                if count == 0:
                    prev_out_spec = out_path_spec
                    df = pd.DataFrame(data=records)
                    compression = None
                    if self.compression != Compression.NONE:
                        compression = {
                            "method": str(self.compression)
                        }
                        if self.compression == Compression.GZIP:
                            out_path_spec = out_path_spec + ".gz"
                        elif self.compression == Compression.BZIP2:
                            out_path_spec = out_path_spec + ".bz2"
                        elif self.compression == Compression.ZIP:
                            out_path_spec = out_path_spec + ".zip"
                        elif self.compression == Compression.XZ:
                            out_path_spec = out_path_spec + ".xz"
                    df.to_csv(out_path_spec, compression=compression)
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)