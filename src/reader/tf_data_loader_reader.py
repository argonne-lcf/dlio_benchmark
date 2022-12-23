"""
   Copyright (c) 2022, UChicago Argonne, LLC
   All Rights Reserved

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
import math
import logging
from time import time, sleep

from src.utils.utility import utcnow, timeit, perftrace
from src.common.enumerations import Shuffle, FormatType
from src.reader.reader_handler import FormatReader
import tensorflow as tf
import numpy as np
import os
from functools import wraps 
import threading
from PIL import Image
import h5py


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = time()
        x = func(*args, **kwargs)
        end = time()
        print(f"loadding: {end - begin}")
        return x
    return wrapper

def read_jpeg(filename):
    return Image.open(filename).resize((224, 224))

def read_png(filename):
    return Image.open(filename).resize((224, 224))

def read_npz(filename):
    data = np.load(filename)
    x = data['x']; y=data['y']
    return np.zeros((224, 224), dtype=np.uint8)

def read_hdf5(f):
    file_h5 = h5py.File(f, 'r')
    d = file_h5['records'][:,:,:]
    l = file_h5['labels'][:]
    return d, l

def read_file(f):
    with open(f, mode='rb') as file: # b is important -> binary
        return file.read()

filereader={
    FormatType.JPEG: read_jpeg, 
    FormatType.PNG: read_png, 
    FormatType.NPZ: read_npz, 
    FormatType.HDF5: read_hdf5, 
}

reader = {'func': None, 'format': None}

class CustomDataset(tf.data.Dataset):
    def _generator(file_name):
        """
        make a generator function that we can query for batches
        """
        t0 = time()
        data = reader['func'](file_name) 
        t1 = time()
        perftrace.event_complete(f"{reader['func'].__name__}", "reader", t0, t1-t0)
        yield np.zeros((224, 224), dtype=np.uint8)

    def __new__(cls, file_name):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.uint8,
            output_shapes=(224, 224),
            args=(file_name,), 
        )
        return dataset

class TFDataLoaderReader(FormatReader):
    """
    Reader for TFRecord files.
    """
    def __init__(self, dataset_type):
        super().__init__(dataset_type)
        self.read_threads = self._args.read_threads
        self.computation_threads = self._args.computation_threads
        self.format = self._args.format
        global reader
        try:
            reader['func'] = filereader[self.format]
            reader['format'] = self.format
        except:
            logging.warning(f"{utcnow()} Unsupported file format {self.format} for data loader, reading as binary files")
            reader['func'] = read_file
            reader['format'] = self.format

        # TODO: DLIO assumes the tfrecord files to contain image/label pairs.
        # This is not always the case, e.g. in BERT, each record is more complex,
        # consisting of 6 lists and a label. Same for DLRM.

    def read(self, epoch_number):
        """
        Sets up the tf data pipeline to read tf record files.
        Called once at the start of every epoch.
        Does not necessarily read in all the data at the start however.
        :param epoch_number:
        """
        # superclass function initializes the file list
        super().read(epoch_number)
        if self.read_threads==0:
            if self._args.my_rank==0:
                logging.warning(f"{utcnow()} `read_threads` is set to be 0 for tf.data loader. We change it to tf.data.AUTOTUNE")
            self.read_threads=tf.data.AUTOTUNE

        options = tf.data.Options()
        options.threading.private_threadpool_size = self.read_threads
        options.threading.max_intra_op_parallelism = self.read_threads
        
        dataset = tf.data.Dataset.from_tensor_slices(self._file_list).with_options(options)
        dataset = dataset.interleave(lambda x: CustomDataset(x), cycle_length=self.read_threads, num_parallel_calls=self.read_threads)
        dataset = dataset.shard(num_shards=self.comm_size, index=self.my_rank)

        if self.sample_shuffle != Shuffle.OFF:
            if self.sample_shuffle == Shuffle.SEED:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size,
                                          seed=self.seed)
            else:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        self._dataset = dataset.batch(self.batch_size, drop_remainder=True)
        
        if self.prefetch_size > 0:
            self._dataset = self._dataset.prefetch(buffer_size=self.prefetch_size)
    def next(self):
        """
        Provides the iterator over tfrecord data pipeline.
        :return: data to be processed by the training step.
        """
        super().next()

        # In tf, we can't get the length of the dataset easily so we calculate it
        if self._debug:
            total = math.floor(self.num_samples*len(self._file_list)/self.batch_size/self.comm_size)
            logging.debug(f"{utcnow()} Rank {self.my_rank} should read {total} batches")

        # The previous version crashed when all workers could not generate the same amount of batches
        # Using the inbuilt tensorflow dataset iteration seems to work fine, was there an advantage of doing it the old way?
        # t1
        for batch in self._dataset:
            yield batch

    def finalize(self):
        pass
