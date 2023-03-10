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

from src.utils.utility import utcnow, perftrace
from src.common.enumerations import DatasetType
from src.reader.reader_handler import FormatReader
import tensorflow as tf

class TFReader(FormatReader):
    """
    Reader for TFRecord files.
    """

    def __init__(self, dataset_type, thread_index):
        super().__init__(dataset_type, thread_index)
        self.read_threads = self._args.read_threads
        self.computation_threads = self._args.computation_threads
        self.count = 0
        # We read the full _file_list here instead of _local_file_list
        # because we will shard the data using the tf.data function

        # TODO: DLIO assumes the tfrecord files to contain image/label pairs.

    @perftrace.event_logging
    def _decode_image(self, parsed_example):
        # Get the image as raw bytes.
        image_raw = parsed_example['image']
        dimension = parsed_example['size']
        # Decode the raw bytes so it becomes a tensor with type.
        image_tensor = tf.io.decode_raw(image_raw, tf.float64)
        # image_tensor = tf.io.decode_image(image_raw)
        resized_image = tf.reshape(image_tensor, [dimension, dimension])
        return tf.pad(resized_image, ((0, self.max_dimension - dimension), (0, self.max_dimension - dimension)))

    # This is not always the case, e.g. in BERT, each record is more complex,
    # consisting of 6 lists and a label. Same for DLRM.
    @perftrace.event_logging
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
        return tf.io.parse_example(serialized=serialized,
                                                    features=features)

    @perftrace.event_logging
    def read(self, epoch_number):
        """
        Sets up the tf data pipeline to read tf record files.
        Called once at the start of every epoch.
        Does not necessarily read in all the data at the start however.
        :param epoch_number:
        """
        # superclass function initializes the file list
        super().read(epoch_number)
        self._dataset = tf.data.TFRecordDataset(filenames=self._file_list, buffer_size=self.transfer_size)
        self._dataset = self._dataset.shard(num_shards=self.comm_size, index=self.my_rank)
        self._dataset = self._dataset.map(self._tf_parse_function, num_parallel_calls=self.computation_threads)
        self._dataset = self._dataset.map(self._decode_image, num_parallel_calls=self.computation_threads)
        self._dataset = self._dataset.batch(self.batch_size, drop_remainder=True)
        self.after_read()

    @perftrace.event_logging
    def next(self):
        """
        Provides the iterator over tfrecord data pipeline.
        :return: data to be processed by the training step.
        """
        super().next()

        # In tf, we can't get the length of the dataset easily so we calculate it
        if self._debug:
            total = math.floor(self.num_samples * len(self._file_list) / self.batch_size / self.comm_size)
            logging.debug(f"{utcnow()} Rank {self.my_rank} should read {total} batches")

        # The previous version crashed when all workers could not generate the same amount of batches
        # Using the inbuilt tensorflow dataset iteration seems to work fine, was there an advantage of doing it the old way?
        total = int(math.ceil(self.get_sample_len() / self.batch_size))
        count = 0
        t0 = time()
        for batch in self._dataset:
            t1 = time()
            perftrace.event_complete(f"TFRecord_{self.dataset_type}_step_{count}",
                                     "tfrecord_reader..next", t0, t1 - t0)
            count += 1
            is_last = 0 if count < total else 1
            yield is_last, batch
            t0 = time()

    @perftrace.event_logging
    def read_index(self, index):
        pass

    @perftrace.event_logging
    def get_sample_len(self):
        return self.samples_per_reader
