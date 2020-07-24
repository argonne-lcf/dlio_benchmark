import math

from src.common.enumerations import Shuffle
from src.reader.reader_handler import FormatReader
import tensorflow as tf

from src.utils.utility import progress


class TFReader(FormatReader):
    def __init__(self):
        super().__init__()
        self.read_threads = self._arg_parser.args.read_threads
        self.computation_threads = self._arg_parser.args.computation_threads

    def _tf_parse_function(self, serialized):
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
        super().next()
        a = iter(self._dataset)
        count = 0
        total = math.ceil(self.num_samples*self.num_files/self.batch_size)
        for i in a:
            progress(count, total, "Reading TFRecord Data")
            count += 1
            yield i
            yield next(a)

    def finalize(self):
        pass
