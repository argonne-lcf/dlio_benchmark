"""
The binary file generator designed for simulating DLRM in DLIO
"""

from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator

import logging
import numpy as np
from numpy import random

from src.utils.utility import progress
from shutil import copyfile

"""
Generator for creating data in BIN format.
"""

class BINGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generate binary data for training and testing.
        """
        super().generate()

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            progress(i+1, self.total_files_to_generate, "Generating Binary Data")
            out_path_spec = self.storage.get_uri(self._file_list[i])
            num_instance = 6548660*14 #4195198976
            X_int = np.random.randint(2557264, size = (num_instance, 13))
            X_cat = np.random.randint(8831335, size = (num_instance, 26))
            y = np.random.randint(2, size=num_instance)
            np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
            np_data = np_data.astype(np.int32)
            if self.compression != Compression.ZIP:
                with open(out_path_spec, 'wb') as output_file:
                    output_file.write(np_data.tobytes())

        # prev_out_spec = ""
        # count = 0
        # for i in range(0, int(self.total_files_to_generate)):
        #     if i % self.comm_size == self.my_rank:
        #         progress(i+1, self.total_files_to_generate, "Generating NPZ Data")
        #         out_path_spec = "{}_{}_of_{}.npz".format(self._file_prefix, i, self.total_files_to_generate)
        #         if self.my_rank == 0:
        #             num_instance = 6548660*14 #4195198976
        #             X_int = np.random.randint(2557264, size = (num_instance, 13))
        #             X_cat = np.random.randint(8831335, size = (num_instance, 26))
        #             y = np.random.randint(2, size=num_instance)
        #             np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
        #             np_data = np_data.astype(np.int32)
        #             prev_out_spec = out_path_spec

        #             if self.compression != Compression.ZIP:
        #                 with open(out_path_spec, 'wb') as output_file:
        #                     output_file.write(np_data.tobytes())
        #             else:
        #                 # np.savez_compressed(out_path_spec, x=records, y=record_labels)
        #                 pass #Need to figure out later
        #             count += 1
        #         else:
        #             if i == int(self.total_files_to_generate)-1:
        #                 num_instance = 6548660*14
        #                 X_int = np.random.randint(2557264, size = (num_instance, 13))
        #                 X_cat = np.random.randint(8831335, size = (num_instance, 26))
        #                 y = np.random.randint(2, size=num_instance)
        #                 np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
        #                 np_data = np_data.astype(np.int32)
        #                 prev_out_spec = out_path_spec

        #                 if self.compression != Compression.ZIP:
        #                     with open(out_path_spec, 'wb') as output_file:
        #                         output_file.write(np_data.tobytes())
        #                 else:
        #                     # np.savez_compressed(out_path_spec, x=records, y=record_labels)
        #                     pass #Need to figure out later
        #                 count += 1
        #             else:
        #                 copyfile(prev_out_spec, out_path_spec)

