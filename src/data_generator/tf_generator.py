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

from src.data_generator.data_generator import DataGenerator
from numpy import random
import tensorflow as tf

from src.utils.utility import progress, utcnow
from shutil import copyfile

import logging
"""
Generator for creating data in TFRecord format.
TODO: Might be interesting / more realistic to add randomness to the file sizes
"""
class TFRecordGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generator for creating data in TFRecord format of 3d dataset.
        """
        super().generate()
        # This create a 2d image representing a single record
        record = random.random((self._dimension, self._dimension))
        record_label = 0
        prev_out_spec =""
        count = 0

        for i in range(0, self.total_files_to_generate):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.total_files_to_generate, "Generating TFRecord Data")
                out_path_spec = "{}_{}_of_{}.tfrecords".format(self._file_prefix, i, self.total_files_to_generate)
                logging.info("{} Generating TFRecord {}".format(utcnow(), out_path_spec))
                # Open a TFRecordWriter for the output-file.
                if count == 0:
                    prev_out_spec = out_path_spec
                    with tf.io.TFRecordWriter(out_path_spec) as writer:
                        for i in range(0, self.num_samples):
                            img_bytes = record.tostring()
                            # TODO: We should adapt this to create realistic datasets for each workload
                            # since only image segmentation uses images
                            data = {
                                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[record_label]))
                            }
                            # Wrap the data as TensorFlow Features.
                            feature = tf.train.Features(feature=data)
                            # Wrap again as a TensorFlow Example.
                            example = tf.train.Example(features=feature)
                            # Serialize the data.
                            serialized = example.SerializeToString()
                            # Write the serialized data to the TFRecords file.
                            writer.write(serialized)
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)
