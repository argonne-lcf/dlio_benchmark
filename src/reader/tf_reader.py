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
from numpy import random

from src.common.constants import MODULE_DATA_READER
from src.utils.utility import utcnow, PerfTrace, event_logging, Profile
from src.common.enumerations import DatasetType
from src.reader.reader_handler import FormatReader
import tensorflow as tf

class TFReader(FormatReader):
    """
    Reader for TFRecord files.
    """
    classname = "TFReader"

    def __init__(self, dataset_type, thread_index, epoch_number):
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_DATA_READER, epoch=epoch_number):
            super().__init__(dataset_type, thread_index, epoch_number)
            self._dataset = None

    def open(self, filename):
        pass

    def close(self, filename):
        pass

    def get_sample(self, filename, sample_index):
        pass

    def _tf_parse_function(self, serialized):
        """
        performs deserialization of the tfrecord.
        :param serialized: is the serialized version using protobuf
        :return: deserialized image and label.
        """
        self.image_idx += 1
        with Profile(name=f"{self.get_sample.__qualname__}.read", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                     image_idx=self.image_idx) as p:
            features = \
                {
                    'image': tf.io.FixedLenFeature([], tf.string),
                    'size': tf.io.FixedLenFeature([], tf.int64)
                }
            parsed_example =  tf.io.parse_example(serialized=serialized, features=features)
            # Get the image as raw bytes.
            image_raw = parsed_example['image']
            dimension = tf.cast(parsed_example['size'], tf.int32).numpy()
            # Decode the raw bytes so it becomes a tensor with type.
            image_tensor = tf.io.decode_raw(image_raw, tf.float64)
            size = dimension * dimension
            p.update(image_size=size)
        with Profile(name=f"{self.get_sample.__qualname__}.resize", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                     image_idx=self.image_idx) as p:
            # image_tensor = tf.io.decode_image(image_raw)
            resized_image = tf.reshape(image_tensor, [dimension, dimension])
            size = self._args.max_dimension * self._args.max_dimension
            p.update(image_size=size)
            return tf.pad(resized_image,
                          ((0, self._args.max_dimension - dimension), (0, self._args.max_dimension - dimension)))

    def next(self):
        with Profile(name=f"{self.next.__qualname__}", cat=MODULE_DATA_READER, ):
            _file_list = self._args.file_list_train if self.dataset_type is DatasetType.TRAIN else self._args.file_list_eval
            batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
            logging.debug(f"{utcnow()} Reading {len(_file_list)} files thread {self.thread_index} rank {self._args.my_rank}")
            self._dataset = tf.data.TFRecordDataset(filenames=_file_list, buffer_size=self._args.transfer_size)
            self._dataset = self._dataset.shard(num_shards=self._args.comm_size, index=self._args.my_rank)
            self._dataset = self._dataset.map(lambda x: tf.py_function(func=self._tf_parse_function, inp=[x], Tout=[tf.float64])
                , num_parallel_calls=self._args.computation_threads)
            self._dataset = self._dataset.batch(batch_size, drop_remainder=True)
            total = math.ceil(len(_file_list) / batch_size)
            step = 1
            with Profile(name=f"{self.next.__qualname__}", cat=MODULE_DATA_READER, ) as lp:
                for batch in self._dataset:
                    lp.update(epoch=self.epoch_number, step=step).flush()
                    step += 1
                    is_last = 0 if step <= total else 1
                    yield is_last, batch[0]
                    lp.reset()

    def read_index(self, index):
        with Profile(name=f"{self.read_index.__qualname__}", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                     image_idx=index):
            return super().read_index(index)

    def finalize(self):
        with Profile(name=f"{self.finalize.__qualname__}", cat=MODULE_DATA_READER, epoch=self.epoch_number):
            return super().finalize()
