from src.data_generator.data_generator import DataGenerator
from numpy import random
import tensorflow as tf

from src.utils.utility import progress
from shutil import copyfile


class TFRecordGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        super().generate()
        record = random.random((self._dimension, self._dimension))
        record_label = 0
        prev_out_spec =""
        count = 0
        for i in range(0, int(self.num_files)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.num_files, "Generating TFRecord Data")
                out_path_spec = "{}_{}_of_{}.tfrecords".format(self._file_prefix, i, self.num_files)
                # Open a TFRecordWriter for the output-file.
                if count == 0:
                    prev_out_spec = out_path_spec
                    with tf.io.TFRecordWriter(out_path_spec) as writer:
                        for i in range(0, self.num_samples):
                            img_bytes = record.tostring()
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
