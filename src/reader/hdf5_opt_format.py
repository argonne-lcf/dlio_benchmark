"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of DLProfile
 DLIO is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
"""

from src.common.enumerations import Shuffle, FileAccess
from src.reader.reader_handler import FormatReader
import h5py
import math
from numpy import random
import tensorflow as tf
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops

class _SleepDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that sleeps before producing each upstream element."""

  def __init__(self, input_dataset, sleep_microseconds):
    self._input_dataset = input_dataset
    self._sleep_microseconds = sleep_microseconds
    variant_tensor = gen_experimental_dataset_ops.sleep_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._sleep_microseconds,
        **self._flat_structure)
    super(_SleepDataset, self).__init__(input_dataset, variant_tensor)


def sleep(sleep_microseconds):
  """Sleeps for `sleep_microseconds` before producing each input element.
  Args:
    sleep_microseconds: The number of microseconds to sleep before producing an
      input element.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _SleepDataset(dataset, sleep_microseconds)

  return _apply_fn

from src.utils.utility import progress

"""
Reader for HDF5 files for training file.
"""
class HDF5Generator(object):

    def __init__(self, hdf5_file,batch_size):
        self._file = hdf5_file
        self._f = None
        self.batch_size = batch_size
        self.step = 1

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['records'].shape[0]
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_examples(self, start_idx, stop_idx):
        with tf.profiler.experimental.Trace('Read', step_num=start_idx/self.batch_size, _r=1):
            images = self._f['records'][start_idx: stop_idx]
        return images





class HDF5Dataset(tf.data.Dataset):
    def _generator(file_name, batch_size, start_idx_, num_events=-1,dimention = 1,transfer_size =-1):
        """
        make a generator function that we can query for batches
        """
        reader = HDF5Generator(file_name, batch_size)
        nevents = reader.openf()
        if num_events == -1:
            num_events = nevents
        if transfer_size == -1:
            num_elements = batch_size
        else:
            num_elements = math.ceil(transfer_size / dimention / dimention)
            if num_elements <= batch_size:
                num_elements = batch_size
            else:
                num_elements = num_elements - (num_elements%batch_size)

        num_yields = math.floor(num_elements / batch_size)
        start_idx, stop_idx,last_event = start_idx_, start_idx_+num_elements,start_idx_+num_events
        step = start_idx_/batch_size
        while True:
            if start_idx >= last_event:
                reader.closef()
                return
            if stop_idx > last_event:
                stop_idx = last_event
            images = reader.get_examples(start_idx, stop_idx)
            for i in range(num_yields):
                step += 1
                yield_images = images[i * batch_size:(i + 1) * batch_size]
                yield step,yield_images
            start_idx, stop_idx = start_idx + num_elements, stop_idx + num_elements



    def __new__(cls, file_name="", batch_size=1, start_idx=0, num_events=-1,dimension = 1, transfer_size =-1):
        features_shape = [batch_size, dimension, dimension]
        step_shape = []
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.int32, tf.dtypes.float32),
            output_shapes=(tf.TensorShape(step_shape), tf.TensorShape(features_shape)),
            args=(file_name, batch_size, start_idx, num_events,dimension,transfer_size,)
        )
        return dataset

class HDF5OptReader(FormatReader):
    def __init__(self):
        super().__init__()
        self.read_threads = self._arg_parser.args.read_threads
        self.computation_threads = self._arg_parser.args.computation_threads

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
                part_size = total_samples_per_rank;
            for file in self._local_file_list:
                for i in range(parallel_threads):
                    local_file_list.append((file,str(i*part_size),str(part_size)))
        options = tf.data.Options()
        options.experimental_threading.private_threadpool_size = 32
        options.experimental_threading.max_intra_op_parallelism = 32
        Dataset = tf.data.Dataset
        print(local_file_list)
        dataset = Dataset.from_tensor_slices(local_file_list).with_options(options)
        if self.transfer_size:
            transfer_size = self.transfer_size
        else:
            transfer_size = -1
        dataset = dataset.interleave(lambda x: HDF5Dataset(x[0],self.batch_size, int(x[1]), int(x[2]),self._dimension,transfer_size,),
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
