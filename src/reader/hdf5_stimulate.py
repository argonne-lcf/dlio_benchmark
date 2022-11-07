"""
   Copyright 2021 UChicago Argonne, LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from src.common.enumerations import Shuffle, FileAccess
from src.reader.reader_handler import FormatReader
import h5py
import math
from numpy import random
import tensorflow as tf
import numpy as np

from src.reader.stimulate.dataset import HDF5Dataset,HDF5Generator
from src.utils.utility import progress

"""
Reader for HDF5 files for training file.
"""

class HDF5StimulateGenerator(HDF5Generator):
    def __init__(self, hdf5_file, batch_size):
        super().__init__(hdf5_file, batch_size)

    def get_nevents(self):
        return self._f['records'].shape[0]

    def get_examples(self, start_idx, stop_idx):
        with tf.profiler.experimental.Trace('Read', step_num=start_idx / self.batch_size, _r=1):
            images = self._f['records'][start_idx: stop_idx]
        return images

class HDF5StimulateReader(FormatReader):
    def __init__(self, dataset_type):
        super().__init__(dataset_type)
        self.read_threads = self._args.read_threads
        self.computation_threads = self._args.computation_threads

    def resize(self, step, image):
        return tf.image.resize(image, (self._dimension, self._dimension))

    def read(self, epoch_number):
        """
        Reading the hdf5 dataset. Here we take just take the filename and they are open during iteration
        :param epoch_number: epoch number for training loop
        """
        super().read(epoch_number)
        packed_array = []
        count = 1
        local_file_list = []
        if FileAccess.MULTI == self.file_access:
            # for multiple file access the whole file would read by each process.
            part_start, part_end = 0, int(math.ceil(self.num_samples/self.batch_size))
            for file in self._local_file_list:
                local_file_list.append((file,str(part_start),str(part_end)))
        else:
            # for shared file access a part of file would be read by each process.
            total_samples_per_rank = int(self.num_samples / self.comm_size)
            part_start, part_end = (int(total_samples_per_rank * self.my_rank / self.batch_size),
                                    int(total_samples_per_rank * (self.my_rank + 1) / self.batch_size))

            if self.read_threads:
                parallel_threads = self.read_threads + 1
                part_size = math.ceil(total_samples_per_rank/parallel_threads)
            else:
                parallel_threads = 1
                part_size = total_samples_per_rank
            for file in self._local_file_list:
                for i in range(parallel_threads):
                    local_file_list.append((file,str(i*part_size),str(part_size)))
        options = tf.data.Options()
        options.experimental_threading.private_threadpool_size = 32
        options.experimental_threading.max_intra_op_parallelism = 32
        Dataset = tf.data.Dataset
        #print(local_file_list)
        dataset = Dataset.from_tensor_slices(local_file_list).with_options(options)
        if self.transfer_size:
            transfer_size = self.transfer_size
        else:
            transfer_size = -1
        features_shape = [self.batch_size, self._dimension, self._dimension]
        step_shape = []
        dataset = dataset.interleave(lambda x: HDF5Dataset("src.reader.hdf5_stimulate.HDF5StimulateGenerator",
                                                           x[0],
                                                           (tf.dtypes.int32, tf.dtypes.float32),
                                                           (tf.TensorShape(step_shape), tf.TensorShape(features_shape)),
                                                           self.batch_size, int(x[1]), int(x[2]),self._dimension,transfer_size,),
                                     cycle_length=self.read_threads,
                                     block_length=1,
                                     num_parallel_calls=self.read_threads)
        #dataset = dataset.cache()
        if self.memory_shuffle != Shuffle.OFF:
            if self.memory_shuffle != Shuffle.SEED:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size,
                                          seed=self.seed)
            else:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        if self.prefetch:
            dataset = dataset.prefetch(buffer_size=self.prefetch_size)

        dataset = dataset.map(self.resize, num_parallel_calls=self.computation_threads)

        self._dataset = dataset

    def next(self):
        """
        This method is called during iteration where a dataset is opened and different regions of the dataset are
        yielded to the training loop
        :return: portion of dataset to be used in step.
        """
        super().next()
        a = iter(self._dataset)
        count = 1
        total = math.ceil(self.num_samples * self.num_files / self.batch_size / self.comm_size)
        for i in a:
            progress(count, total, "Reading HDF5 Optimized Data")
            count += 1
            yield i
            if count > total:
                break

    def finalize(self):
        pass
