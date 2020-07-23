import h5py
from numpy import random

from src.data_generator.data_generator import DataGenerator


class HDF5Generator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        super().generate()
        records = random.random((self._dimension, self._dimension, self.num_samples))
        record_labels = [0] * self.num_samples
        for i in range(0, int(self.num_files)):
            out_path_spec = "{}_{}_of_{}.h5".format(self._file_prefix, i, self.num_files)
            hf = h5py.File(out_path_spec, 'w')
            hf.create_dataset('records', data=records)
            hf.create_dataset('labels', data=record_labels)
            hf.close()
