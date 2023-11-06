#!/usr/bin/env python
from dlio_benchmark.utils.utility import Profile, PerfTrace
import argparse
import numpy as np
import os
import h5py
parser = argparse.ArgumentParser(
                    prog='DLIO testing',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--log_dir', default="./pfw_logs", type=str, help="The log directory to save to the tracing")
parser.add_argument("--data_dir", default="./data", type=str, help="The directory to save and load data")
parser.add_argument("--format", default="npz", type=str, help="format of the file")
parser.add_argument("--num_files", default=1, type=int, help="Number of files")
parser.add_argument("--niter", default=1, type=int, help="Number of iterations for the experiment")
parser.add_argument("--record_size", default=1048576, type=int, help="size of the record to be written to the file")
args = parser.parse_args()
os.makedirs(f"{args.log_dir}/{args.format}", exist_ok=True)
os.makedirs(f"{args.data_dir}/{args.format}", exist_ok=True)
dlp = Profile("dlio")

PerfTrace.initialize_log(f"{args.log_dir}/{args.format}", f"{args.data_dir}/{args.format}")
from PIL import Image
import cv2

class IOHandler:
    def __init__(self, format):
        self.format = format
    def read(self, filename):
        if (self.format=="jpeg" or self.format=="png"):
            return np.asarray(Image.open(filename))
        if (self.format=="npz"):
            return np.load(filename)
        if (self.format=="hdf5"):
            fd = h5py.File(filename, 'r')
            x = fd['x'][:]  
            fd.close()
    def write(self, filename, a):
        if (self.format=="jpeg" or self.format=="png"):
            #cv2.imwrite(filename, a)
            im = Image.fromarray(a)
            #im.show()
            im.save(filename)
        if (self.format=="npz"):
            with open(filename, 'wb') as f:
                np.save(f, a)
        if (self.format=="hdf5"):
            fd = h5py.File(filename, 'w')
            fd.create_dataset("x", data=a)
            fd.close()
io = IOHandler(args.format)
# Writing data
data = np.ones((args.record_size, 1), dtype=np.uint8)

def data_gen(data):
    print(data)
    for i in dlp.iter(range(args.num_files)):
        io.write(f"{args.data_dir}/{args.format}/{i}-of-{args.num_files}.{args.format}", data)

def read_data(epoch):        
    for i in dlp.iter(range(args.num_files)):
        d = io.read(f"{args.data_dir}/{args.format}/{i}-of-{args.num_files}.{args.format}")

data_gen(data)
for n in range(args.niter):
    read_data(n)


PerfTrace.get_instance().finalize()




