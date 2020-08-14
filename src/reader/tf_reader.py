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

from src.common.enumerations import Shuffle
from src.reader.reader_handler import FormatReader
import tensorflow as tf

from src.utils.utility import progress


class TFReader(FormatReader):
    """
    Reader for TFRecord files.
    """
    def __init__(self):
        super().__init__()
        self.read_threads = self._arg_parser.args.read_threads
        self.computation_threads = self._arg_parser.args.computation_threads

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

    def read(self, epoch_number):
        """
        Sets up the tf data pipeline to read tf record files.
        :param epoch_number:
        """
        super().read(epoch_number)
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
        self._dataset = dataset.batch(self.batch_size)

    def next(self):
        """
        Provides the iterator over tfrecord data pipeline.
        :return: data to be processed by the training step.
        """
        super().next()
        a = iter(self._dataset)
        count = 1
        total = math.ceil(self.num_samples*self.num_files/self.batch_size/self.comm_size)
        for i in a:
            progress(count, total, "Reading TFRecord Data")
            count += 1
            yield i
            yield next(a)
            if count > total:
                break

    def finalize(self):
        pass
