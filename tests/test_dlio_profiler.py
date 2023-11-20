#!/usr/bin/env python
from dlio_profiler.logger import fn_interceptor as Profile
from dlio_profiler.logger import dlio_logger as PerfTrace
import argparse
import numpy as np
import os
import h5py
import tensorflow as tf
import glob
parser = argparse.ArgumentParser(
                    prog='DLIO testing',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--log_dir', default="./pfw_logs", type=str, help="The log directory to save to the tracing")
parser.add_argument("--data_dir", default="./data", type=str, help="The directory to save and load data")
parser.add_argument("--format", default="npz", type=str, help="format of the file")
parser.add_argument("--num_files", default=16, type=int, help="Number of files")
parser.add_argument("--niter", default=4, type=int, help="Number of iterations for the experiment")
parser.add_argument("--record_size", default=1048576, type=int, help="size of the record to be written to the file")
args = parser.parse_args()
os.makedirs(f"{args.log_dir}/{args.format}", exist_ok=True)
os.makedirs(f"{args.data_dir}/{args.format}", exist_ok=True)
dlp = Profile("dlio")

PerfTrace.initialize_log(f"{args.log_dir}/{args.format}", 
                         f"{args.data_dir}")
#f"{args.data_dir}/{args.format}")
from PIL import Image
import cv2

@dlp.log
def tf_parse_image(serialized):
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
    return image_tensor

@dlp.log
def tf_write_image(filename, record, size):
    with tf.io.TFRecordWriter(filename) as writer:
        img_bytes = record.tobytes()
        data = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
            'size': tf.train.Feature(int64_list=tf.train.Int64List(value=[size]))
        }
        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=data)
        # Wrap again as a TensorFlow Example.
        example = tf.train.Example(features=feature)
        # Serialize the data.
        serialized = example.SerializeToString()
        # Write the serialized data to the TFRecords file.
        writer.write(serialized)

class IOHandler:
    def __init__(self, format, path="./"):
        self.format = format
        self._path = path
        self._flist = glob.glob(f"{path}/*{self.format}")
        if (self.format=="tfrecord"):
            self._dataset = tf.data.TFRecordDataset(filenames=self._flist)
            self._dataset = self._dataset.map(
                lambda x: tf.py_function(func=tf_parse_image, inp=[x], Tout=[tf.uint8]))
    @dlp.log            
    def read(self, filename):
        if (self.format=="jpeg" or self.format=="png"):
            return np.asarray(Image.open(filename))
        if (self.format=="npz"):
            x=np.load(filename, allow_pickle=True)
            return x
        if (self.format=="hdf5"):
            fd = h5py.File(filename, 'r')
            x = fd['x'][:]  
            fd.close()
            return x
        
    @dlp.log
    def write(self, filename, a):
        if (self.format=="jpeg" or self.format=="png"):
            #cv2.imwrite(filename, a)
            im = Image.fromarray(a)
            #im.show()
            im.save(filename)
        if (self.format=="npz"):
            with open(filename, 'wb') as f:
                np.save(f, a, allow_pickle=True)
        if (self.format=="hdf5"):
            fd = h5py.File(filename, 'w')
            fd.create_dataset("x", data=a)
            fd.close()
        if (self.format=="tfrecord"):
            tf_write_image(filename, a, a.shape[0])

    @dlp.log
    def train_epoch(self):
        if (self.format!="tfrecord"):
            for f in dlp.iter(self._flist):
                x = self.read(f)
        else:
            for f in dlp.iter(self._dataset):
                f = 0
            
io = IOHandler(args.format, f"{args.data_dir}/{args.format}")
# Writing data
data = np.ones((args.record_size, 1), dtype=np.uint8)
def data_generator(data):
    for i in range(args.num_files):
        print(f"Generating {args.data_dir}/{args.format}/{i}-of-{args.num_files}.{args.format}")
        io.write(f"{args.data_dir}/{args.format}/{i}-of-{args.num_files}.{args.format}", data)

def train(epoch):        
    io.train_epoch()

def main():    
    data_generator(data)
    for n in range(args.niter):
        train(n)

if __name__=="__main__":
    main()
