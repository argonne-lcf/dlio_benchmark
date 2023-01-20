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

from src.data_generator.data_generator import DataGenerator
from numpy import random
import tensorflow as tf

from src.utils.utility import progress, utcnow
from shutil import copyfile

import logging
class TFRecordGenerator(DataGenerator):
    """
    Generator for creating data in TFRecord format.
    """
    def __init__(self):
        super().__init__()
    def generate(self):
        """
        Generator for creating data in TFRecord format of 3d dataset.
        TODO: Might be interesting / more realistic to add randomness to the file sizes.
        TODO: Extend this to create accurate records for BERT, which does not use image/label pairs.
        """
        super().generate()
        # This creates a 2D image representing a single record
        record_label = 0
        for i in range(self.my_rank, self.total_files_to_generate, self.comm_size):
            progress(i+1, self.total_files_to_generate, "Generating TFRecord Data")
            out_path_spec = self.storage.get_uri(self._file_list[i])
            if (self._dimension_stdev>0):
                dim1, dim2 = [max(int(d), 0) for d in random.normal( self._dimension, self._dimension_stdev, 2)]
            else:
                dim1 = dim2 = self._dimension
            record = random.random((dim1, dim2))
            # Open a TFRecordWriter for the output-file.
            with tf.io.TFRecordWriter(out_path_spec) as writer:
                for i in range(0, self.num_samples):
                    # This creates a 2D image representing a single record
                    record = random.random((self._dimension, self._dimension))
                    img_bytes = record.tobytes()
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
