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

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.tf_base_reader import TFBaseReader
from dlio_benchmark.utils.utility import utcnow, PerfTrace, Profile
from dlio_benchmark.common.enumerations import DatasetType
import tensorflow as tf

dlp = Profile(MODULE_DATA_READER)


class TFTFRecordReader(TFBaseReader):
    @dlp.log_init
    def __init__(self, dataset_type):
        super().__init__(dataset_type)

    @dlp.log
    def parse_image(self, serialized):
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
        parsed_example = tf.io.parse_example(serialized=serialized, features=features)
        # Get the image as raw bytes.
        image_raw = parsed_example['image']
        dimension = tf.cast(parsed_example['size'], tf.int32).numpy()
        # Decode the raw bytes so it becomes a tensor with type.
        image_tensor = tf.io.decode_raw(image_raw, tf.uint8)
        size = dimension * dimension
        dlp.update(image_size=size)
        # image_tensor = tf.io.decode_image(image_raw)
        return image_tensor

    @dlp.log
    def _load(self):
        logging.debug(
            f"{utcnow()} Reading {len(self.file_list)} files rank {self._args.my_rank}")
        dataset = tf.data.TFRecordDataset(filenames=self.file_list, buffer_size=self._args.transfer_size)
        dataset = dataset.shard(num_shards=self._args.comm_size, index=self._args.my_rank)
        dataset = dataset.map(
            lambda x: tf.py_function(func=self.parse_image, inp=[x], Tout=[tf.uint8])
            , num_parallel_calls=self._args.computation_threads)
        return dataset

    @dlp.log
    def finalize(self):
        pass