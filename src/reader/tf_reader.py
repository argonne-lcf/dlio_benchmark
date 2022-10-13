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
import math
import logging

from src.utils.utility import utcnow
from src.common.enumerations import Shuffle
from src.reader.reader_handler import FormatReader
import tensorflow as tf


class TFReader(FormatReader):
    """
    Reader for TFRecord files.
    """
    def __init__(self):
        super().__init__()
        self.read_threads = self._arg_parser.args.read_threads
        self.computation_threads = self._arg_parser.args.computation_threads

    # TODO: DLIO assumes the tfrecord files to contain image/label pairs.
    # This is not always the case, e.g. in BERT, each record is more complex,
    # consisting of 6 lists and a label. Same for DLRM. 
    def _tf_parse_function(self, serialized):
        """
        performs deserialization of the tfrecord.
        :param serialized: is the serialized version using protobuf
        :return: deserialized image and label.
        """
        features = \
            {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)
            }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)
        # Get the image as raw bytes.
        dimention = int(math.sqrt(self.record_size))
        image_shape = tf.stack([dimention, dimention, 1])
        image_raw = parsed_example['image']
        label = tf.cast(parsed_example['label'], tf.float32)
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.io.decode_raw(image_raw, tf.uint8)
        d = image, label
        return d

    def read(self, epoch_number, do_eval=False):
        """
        Sets up the tf data pipeline to read tf record files.
        Called once at the start of every epoch.
        :param epoch_number:
        """
        # superclass function initializes the file list
        super().read(epoch_number, do_eval)

        dataset = tf.data.TFRecordDataset(filenames=self._local_file_list,
                                          buffer_size=self.transfer_size,
                                          num_parallel_reads=self.read_threads)

        dataset = dataset.map(self._tf_parse_function, num_parallel_calls=self.computation_threads)

        if self.memory_shuffle != Shuffle.OFF:
            if self.memory_shuffle != Shuffle.SEED:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size,
                                          seed=self.seed)
            else:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        if self.prefetch:
            dataset = dataset.prefetch(buffer_size=self.prefetch_size)
        self._dataset = dataset.batch(self.batch_size, drop_remainder=True)


    def next(self):
        """
        Provides the iterator over tfrecord data pipeline.
        :return: data to be processed by the training step.
        """
        super().next()
        dataset = self._dataset

        # In tf, we can't get the length of the dataset easily so we calculate it
        if self._debug:
            total = math.ceil(self.num_samples*self._local_file_list_size/self.batch_size/self.comm_size)
            logging.debug(f"{utcnow()} Rank {self.my_rank} should read {total} batches")

        # The previous version crashed when all workers could not generate the same amount of batches
        # Using the inbuilt tensorflow dataset iteration seems to work fine, was there an advantage of doing it the old way?
        for batch in dataset:
            yield batch

    def finalize(self):
        pass
