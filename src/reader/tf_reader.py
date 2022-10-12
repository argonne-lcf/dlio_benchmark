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
        param: epoch_number
        """
        # superclass function initializes the file list
        super().read(epoch_number, do_eval)

        if do_eval:
            dataset = tf.data.TFRecordDataset(filenames=self._local_eval_file_list,
                                    buffer_size=self.transfer_size,
                                    num_parallel_reads=self.read_threads)
            batch_size = self.batch_size_eval
        else:
            dataset = tf.data.TFRecordDataset(filenames=self._local_train_file_list,
                                    buffer_size=self.transfer_size,
                                    num_parallel_reads=self.read_threads)
            batch_size = self.batch_size

        dataset = dataset.map(self._tf_parse_function, num_parallel_calls=self.computation_threads)

        if self.memory_shuffle != Shuffle.OFF:
            if self.memory_shuffle != Shuffle.SEED:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size,
                                        seed=self.seed)
            else:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        if self.prefetch:
            dataset = dataset.prefetch(buffer_size=self.prefetch_size)

        self._dataset = dataset.batch(batch_size, drop_remainder=True)

        # # We're evaluating, load the eval dataset
        # if do_eval:
        #     dataset_eval = tf.data.TFRecordDataset(filenames=self._local_eval_file_list,
        #                                     buffer_size=self.transfer_size,
        #                                     num_parallel_reads=self.read_threads)
        #     dataset_eval = dataset_eval.map(self._tf_parse_function, num_parallel_calls=self.computation_threads)
        #     # No shuffling for eval set for now
        #     if self.prefetch:
        #         dataset_eval = dataset_eval.prefetch(buffer_size=self.prefetch_size)
        #     self._dataset_eval = dataset_eval.batch(self.batch_size, drop_remainder=True)
        # else:
        #     dataset_train = tf.data.TFRecordDataset(filenames=self._local_train_file_list,
        #                                     buffer_size=self.transfer_size,
        #                                     num_parallel_reads=self.read_threads)
        #     dataset_train = dataset_train.map(self._tf_parse_function, num_parallel_calls=self.computation_threads)

        #     if self.memory_shuffle != Shuffle.OFF:
        #         if self.memory_shuffle != Shuffle.SEED:
        #             dataset_train = dataset_train.shuffle(buffer_size=self.shuffle_size,
        #                                     seed=self.seed)
        #         else:
        #             dataset_train = dataset_train.shuffle(buffer_size=self.shuffle_size)
        #     if self.prefetch:
        #         dataset_train = dataset_train.prefetch(buffer_size=self.prefetch_size)
        #     self._dataset_train = dataset_train.batch(self.batch_size, drop_remainder=True)



    def next(self, do_eval=False):
        """
        Provides the iterator over tfrecord data pipeline.
        :return: data to be processed by the training step.
        """
        super().next()

        dataset = self._dataset

        # In tf, we can't get the length of the dataset easily so we calculate it
        if self._debug:
            if do_eval:
                total = math.ceil(self.num_samples*self._local_eval_file_list_size/self.batch_size_eval/self.comm_size)
            else:
                total = math.ceil(self.num_samples*self._local_train_file_list_size/self.batch_size/self.comm_size)

            logging.debug(f"{utcnow()} Rank {self.my_rank} should read {total} batches")

        # The previous version crashed when all workers could not generate the same amount of batches
        # Using the inbuilt tensorflow dataset iteration seems to work fine, was there an advantage of doing it the old way?
        for batch in dataset:
            yield batch

    def finalize(self):
        pass
