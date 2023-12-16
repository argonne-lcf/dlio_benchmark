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
import os
from subprocess import call

from dlio_benchmark.data_generator.data_generator import DataGenerator
import numpy as np
import tensorflow as tf
from dlio_profiler.logger import fn_interceptor as Profile

from dlio_benchmark.utils.utility import progress, utcnow
from shutil import copyfile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)
import logging
class TFRecordGenerator(DataGenerator):
    """
    Generator for creating data in TFRecord format.
    """
    def __init__(self):
        super().__init__()
    @dlp.log
    def generate(self):
        """
        Generator for creating data in TFRecord format of 3d dataset.
        TODO: Might be interesting / more realistic to add randomness to the file sizes.
        TODO: Extend this to create accurate records for BERT, which does not use image/label pairs.
        """
        super().generate()
        np.random.seed(10)
        # This creates a 2D image representing a single record
        record_label = 0
        for i in dlp.iter(range(self.my_rank, self.total_files_to_generate, self.comm_size)):
            progress(i+1, self.total_files_to_generate, "Generating TFRecord Data")
            out_path_spec = self.storage.get_uri(self._file_list[i])
            dim1, dim2 = self.get_dimension()
            # Open a TFRecordWriter for the output-file.
            with tf.io.TFRecordWriter(out_path_spec) as writer:
                for i in range(0, self.num_samples):
                    # This creates a 2D image representing a single record
                    record = np.random.randint(255, size=(dim1, dim2), dtype=np.uint8)
                    img_bytes = record.tobytes()
                    data = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                        'size': tf.train.Feature(int64_list=tf.train.Int64List(value=[self._dimension]))
                    }
                    # Wrap the data as TensorFlow Features.
                    feature = tf.train.Features(feature=data)
                    # Wrap again as a TensorFlow Example.
                    example = tf.train.Example(features=feature)
                    # Serialize the data.
                    serialized = example.SerializeToString()
                    # Write the serialized data to the TFRecords file.
                    writer.write(serialized)
            tfrecord2idx_script = "tfrecord2idx"
            folder = "train"
            if "valid" in out_path_spec:
                folder = "valid"
            index_folder = f"{self._args.data_folder}/index/{folder}"
            filename = os.path.basename(out_path_spec)
            self.storage.create_node(index_folder, exist_ok=True)
            tfrecord_idx = f"{index_folder}/{filename}.idx"
            if not os.path.isfile(tfrecord_idx):
                call([tfrecord2idx_script, out_path_spec, tfrecord_idx])
        np.random.seed()
