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
from time import time

from src.utils.utility import utcnow, PerfTrace,event_logging
from src.common.enumerations import DatasetType
from src.reader.reader_handler import FormatReader
import tensorflow as tf
MY_MODULE = "reader"

class TFReader(FormatReader):
    """
    Reader for TFRecord files.
    """
    classname = "TFReader"

    def __init__(self, dataset_type, thread_index, epoch_number):
        t0 = time()
        super().__init__(dataset_type, thread_index, epoch_number)
        self._dataset = None
        t1 = time()
        PerfTrace.get_instance().event_complete(f"{self.__init__.__qualname__}", MY_MODULE, t0, t1 - t0)

    def open(self, filename):
        pass

    def close(self, filename):
        pass

    def get_sample(self, filename, sample_index):
        pass

    @event_logging(module=MY_MODULE)
    def _decode_image(self, parsed_example):
        # Get the image as raw bytes.
        image_raw = parsed_example['image']
        dimension = parsed_example['size']
        # Decode the raw bytes so it becomes a tensor with type.
        image_tensor = tf.io.decode_raw(image_raw, tf.float64)
        # image_tensor = tf.io.decode_image(image_raw)
        resized_image = tf.reshape(image_tensor, [dimension, dimension])
        return tf.pad(resized_image, ((0, self._args.max_dimension - dimension), (0, self._args.max_dimension - dimension)))

    @event_logging(module=MY_MODULE)
    def _tf_parse_function(self, serialized):
        """
        performs deserialization of the tfrecord.
        :param serialized: is the serialized version using protobuf
        :return: deserialized image and label.
        """
        features = \
            {
                'image': tf.io.FixedLenFeature([], tf.string),
                'size': tf.io.FixedLenFeature([], tf.int64)
            }
        return tf.io.parse_example(serialized=serialized, features=features)

    @event_logging(module=MY_MODULE)
    def next(self):
        _file_list = self._args.file_list_train if self.dataset_type is DatasetType.TRAIN else self._args.file_list_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        logging.debug(f"{utcnow()} Reading {len(_file_list)} files thread {self.thread_index} rank {self._args.my_rank}")
        self._dataset = tf.data.TFRecordDataset(filenames=_file_list, buffer_size=self._args.transfer_size)
        self._dataset = self._dataset.shard(num_shards=self._args.comm_size, index=self._args.my_rank)
        self._dataset = self._dataset.map(self._tf_parse_function, num_parallel_calls=self._args.computation_threads)
        self._dataset = self._dataset.map(self._decode_image, num_parallel_calls=self._args.computation_threads)
        self._dataset = self._dataset.batch(batch_size, drop_remainder=True)
        total = math.ceil(len(_file_list) / batch_size)
        count = 0
        t0 = time()
        for batch in self._dataset:
            t1 = time()
            PerfTrace.get_instance().event_complete(f"{self.next.__qualname__}.next", MY_MODULE, t0, t1 - t0)
            count += 1
            is_last = 0 if count < total else 1
            yield is_last, batch
            t0 = time()

    @event_logging(module=MY_MODULE)
    def read_index(self, index):
        return super().read_index(index)

    @event_logging(module=MY_MODULE)
    def finalize(self):
        return super().finalize()
