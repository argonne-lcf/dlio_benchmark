import h5py
from numpy import random
import math

from src.data_generator.data_generator import DataGenerator
from src.utils.utility import progress


class HDF5Generator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.chunk_size = self._arg_parser.args.chunk_size

    def generate(self):
        super().generate()
        records = random.random((self.num_samples, self._dimension, self._dimension))
        record_labels = [0] * self.num_samples
        for i in range(0, int(self.num_files)):
            progress(i+1, self.num_files, "Generating HDF5 Data")
            out_path_spec = "{}_{}_of_{}.h5".format(self._file_prefix, i, self.num_files)
            hf = h5py.File(out_path_spec, 'w')
            chunk_dimension = int(math.sqrt(self.chunk_size))
            hf.create_dataset('records', data=records, chunks=(1, chunk_dimension, chunk_dimension))
            hf.create_dataset('labels', data=record_labels)
            hf.close()
