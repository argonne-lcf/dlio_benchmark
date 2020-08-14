# DLIO Benchmark
This is repository for a I/O benchmark which represents Scientific Deep Learning Workloads. DLIO benchmark is aimed at emulating the behavior of scientific deep learning applications, as described in the previous section. The benchmark is delivered as an executable that can be configured for various I/O patterns. It uses a modular design to incorporate more data formats, datasets, and configuration parameters. It emulates modern scientific deep learning applications using Benchmark Runner, Data Generator, Format Handler, and I/O Profiler modules. These modules utilize state-of-the-art design patterns to build a transparent and extensible framework. The DLIO benchmark has been designed with the following goals in mind.

## DLIO features include:

- Easy-to-use and highly configurable argument list to emulate any DL application's I/O behavior.
- Fast prototyping through highly modular components to enhance the benchmark with more data formats.
- Full transparency over emulation of I/O access with logging at different levels.
- Easy to use data generator to test the performance of different data layouts and its impact on the I/O performance.
- Compatible with modern profiling tools such as Tensorboard and Darshan to extract and analyze I/O behavior.

## Usage
```bash
# Get the options available using
python ./dlio_benchmark.py -h

# Example option list
DATA_DIR=~/dlio_datasets/temp
OPTS=(-f tfrecord -fa multi -nf 1024 -sf 1024 -df ${DATA_DIR} -rl 262144 -gd 1 -k 1)
python ./dlio_benchmark.py ${OPTS[@]}

# To only generate data
DATA_DIR=~/dlio_datasets/temp
OPTS=(-f tfrecord -fa multi -nf 1024 -sf 1024 -df ${DATA_DIR} -rl 262144 -gd 1 -go 1 -k 1)
python ./dlio_benchmark.py ${OPTS[@]}

# To run on already generated data
DATA_DIR=~/dlio_datasets/temp
OPTS=(-f tfrecord -fa multi -nf 1024 -sf 1024 -df ${DATA_DIR} -rl 262144 -gd 0 -k 1)
python ./dlio_benchmark.py ${OPTS[@]}
```

## Installation

### Requirements
- horovod~=0.19.5
- tensorflow~=2.2.0
- numpy~=1.19.1
- h5py~=2.10.0

### Installations Instructions
To install VaniDL, the easiest way is to run

For the bleeding edge version (recommended):
```bash
pip install git+https://github.com/hariharan-devarajan/dlio_benchmark
```

For the latest stable version:
```bash
pip install dlio_benchmark
```

Otherwise, you can also install from source by running (from source folder):
```bash
python setup.py install
```
On Theta
```bash
module load DLIO
```

## Contributions
This is the first release of DLIO, if you find any bug, please report it in the GitHub issues section.

Improvements and requests for new features are more than welcome! Do not hesitate to twist and tweak DLIO, and send pull-requests.

## Remaining feature list
- Add argument validations
    - Shared should use one file
    - multiple should have atleast files = nranks
- Add Collective reading
    - create g groups within communicator (configurable when g == # of processes then there is no collective)
    - randomly select 1 process from all groups which will read and then send read data to other processes. in the group
- Add Computations
    - Synchronous: Add computation cycles after reading data (busy waiting).
    - Asynchronous: Add I/O on a different thread to overlap with previous compute.
        - use a queue
            - io thread uses queue to read next element.
            - main thread (compute) puts element to queue and then goes to compute.

## License
MIT License



    