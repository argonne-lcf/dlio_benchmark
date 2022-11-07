# Deep Learining I/O (DLIO) Benchmark

## Overview

This is repository for a I/O benchmark which represents Deep Learning Workloads. DLIO benchmark is aimed at emulating the I/O behavior of deep learning applications. The benchmark is delivered as an executable that can be configured for various I/O patterns. It uses a modular design to incorporate more data loaders, data formats, datasets, and configuration parameters. It emulates modern deep learning applications using Benchmark Runner, Data Generator, Format Handler, and I/O Profiler modules. 

### DLIO features include 
* Easy-to-use and highly configurable argument list to emulate any DL application's I/O behavior.
* Fast prototyping through highly modular components to enhance the benchmark with more data formats.
* Full transparency over emulation of I/O access with logging at different levels.
* Easy to use data generator to test the performance of different data layouts and its impact on the I/O performance.
* Compatible with modern profiling tools such as Tensorboard and Darshan to extract and analyze I/O behavior

### Example supported workloads

- UNET3D: 3D Medical Image Segmentation 
  - Reference Implementation: https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch
  - Framework: PyTorch
  - Dataset: `.npz` format image files containing a single sample.
  - Trains over multiple epochs, performs evaluation on a held-out test set periodically.

- BERT: A Large Language Model
  - Reference Implementation: https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert
  - Framework: Tensorflow
  - Dataset: Multiple `tfrecord` files containing many samples each.
  - Trains in a single epoch, performs periodic checkpointing of its parameters.


- DLRM: Deep Learning Recommendation Model (Work in Progress)
  - Reference Implementation: https://github.com/facebookresearch/dlrm/tree/6d75c84d834380a365e2f03d4838bee464157516
  - Framework: PyTorch
  - Dataset: a single large file containing all the training records, and a seocnd one with the evaluation records
  - Trains in a single epoch, and performs periodic evaluations.

## Installation

### In conda environment
```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt 
export PYTHONPATH=$PWD/src:$PYTHONPATH
python ./src/dlio_benchmark.py -h
```

### In Docker container
You must have docker installed on your system. Refer to https://docs.docker.com/get-docker/ for instructions.

Clone the repository.
```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
```

Build the docker image.
```bash
sudo docker build -t dlio:<tag> .
```
The image name can be anything you want. The `<tag>` field is optionnal and serves to differentiate versions of an image.

<br>

## Running the benchmark

A DLIO run is split in 3 phases: 
- Generate synthetic data DLIO will use
- Run the benchmark using the previously generated data
- Post-process the results to generate a report

`start_dlio.sh` provides a convenient way to run these steps, by copying the given scripts to the container and running them within. This way, we don't have to rebuild the image every time we modify the scripts. The script will flush the caches on the host between data-generation and running the benchmark, as well as start `iostat` to gather device-level I/O information. 

```
$ sudo ./start_dlio.sh --help
Usage: ./start_dlio.sh [OPTIONS] -- [EXTRA ARGS]
Convenience script to launch the DLIO benchmark and container.

The given data-generation and run scripts will be launched within the container, flushing the caches between them.
If no data-generation script is given, the data is assumed to have previously been generated in the data directory.
If no run-script is given, an interactive session to the container will be started instead.

Options:
  -h, --help                    Print this message.
  -dd, --data-dir               Directory where the training data is read and generated. ./data by default.
  -od, --output-dir             Output directory for log and checkpoint files. ./output by default.
  -bd, --device                 An I/O device to trace. Can be passed multiple times.
  -im, --image-name             Name of the docker image to launch the container from. Defaults to 'dlio:latest'.
  -c, --container-name          Name to give the docker container. Defaults to dlio.
  -dgs, --datagen-script        Script to generate the data for this run. If empty, data will be assumed to exist in data-dir.
  -rs, --run-script             Script used to launch DLIO within the container.
  -pps, --postproc-script       Post-Porcessing script to generate a report from the DLIO output.
  -it, --interactive            Pass withouth a value. Will launch an interactive session to the container. Gets activated if no run-script or post-processing scripts are given.

Extra args:
  Any extra arguments passed after after '--' will be passed as is to the DLIO launch script.
```

We have included some scripts to emulate the MLCommons UNET3D and BERT workloads under `workloads/`.
The simulated compute times for each of these workloads was measured on a DGX-1 A100 system, and will have to be changed to simulate the behaviour of other systems. 

Make sure to remove or rename the data-directory between runs of different workloads or else the run will fail.

### UNET3D

To run the UNET3D simulation:
```
sudo ./start_dlio.sh -im <image:tag> -dgs workloads/UNET3D/datagen.sh -rs workloads/UNET3D/run.sh -pps workloads/UNET3D/postproc.sh -bd <dev-to-trace>
```

You can include multiple `-bd` options to trace multiple devices.

### BERT

To run the BERT simulation:
```
sudo ./start_dlio.sh -im <image:tag> -dgs workloads/BERT/datagen.sh -rs workloads/BERT/run.sh -pps workloads/BERT/postproc.sh -bd <dev-to-trace>
```
### DLRM
Work in progress.

<br>

## Command line options for DLIO

```
$ python3 src/dlio_benchmark.py --help
  The configuration can be specified by a yaml config file.

  A complete list of config options are: 

  workflow:
    generate_data: whether to generate data
    train: whether to perform training 
    debug: whether to turn on debugging
    profiling: whether to enable profiling within benchmark
  framework: specifying the framework to use [tensorflow | pytorch]
  dataset:
    record_length: size of sample in bytes
    format: the format of the file that the dataset is stored [hdf5|png|jepg|csv...]
    num_files_train: number of files for training dataset
    num_files_val:  number of files for validation dataset
    num_samples_per_file:  number of samples per file
    data_dir: the directory that the dataset is stored
    batch_size: batch size for the training dataset
    batch_size_eval: batch size fo the validation dataset 
    file_prefix: the prefix of the dataset files 
    compression: compression to use
    compression_level: Level of compression for GZIP
    chunking: whether to use chunking in generating HDF5 datasets
  data_loader: 
    data_loader: the data loader to use [tensorflow|pytorch]
    read_threads: number of threads to load the dataset
    computation_threads:  number of threads for preprocessing the data
    prefetch: whether to prefetch the data
    prefetch_size: the buffer size for prefetch
    read_shuffle: whether to shuffle the dataset
    shuffle_size: the shuffle buffer size in byte
    read_type: whether it is ON_DEMAND or MEMORY (stored in the memory)
    file_access: multiple files or shared file access
    transfer_size: transfer size for tensorflow data loader
  train:
    n_epochs: number of epochs for training
    computation_time: simulated training time (in seconds) for each training step
    eval_time: simulated evaluation time (in seconds) for each step
    total_training_steps:  total number of traning steps. If this is set, n_epochs will be ignored
    seed_change_epoch: whether to change the random seed after each epoch 
    eval_after_epoch: start evaluation after eval_after_epoch epochs
    do_eval: whether to do evaluation
    seed: the random seed
  checkpoint: 
    do_checkpoint:  whether to do checkpoint
    checkpoing_after_epoch: start checkpointing after certain number of epochs specified 
    epochs_between_checkpoints: performing one checkpointing per certain number of epochs specified 
    output_folder: the output folder for checkpointing 
    model_size:  the size of the model in bytes

  You can override everything in a command line, for example:
  python src/dlio_benchmark.py framework=tensorflow
```

## Current Limitations and Future Work

* DLIO currently assumes the samples to always be 2D images, even though one can set the size of each sample through ```--record_length```. We expect the shape of the sample to have minimal impact to the I/O itself. This yet to be validated for case by case perspective. We plan to add option to allow specifying the shape of the sample. 

* We assume the data/label pairs are stored in the same file. Storing data and labels in separate files will be supported in future.

* File format support: we only support tfrecord, hdf5, npz, csv, jpg, jpeg formats. Other data formats can be extended. 

* Data Loader support: we support reading datasets using TensorFlow tf.data data loader, PyTorch DataLoader, and a set of custom data readers implemented in ./reader. For TensorFlow tf.data data loader, PyTorch DataLoader  
  - We have complete support for tfrecord format in TensorFlow data loader. 
  - For npz, jpg, jpeg, hdf5, we currently only support one sample per file case. In other words, each sample is stored in an independent file. Multiple samples per file case will be supported in future. 

## How to contribute 
TBD

## Citation
```
@article{devarajan2021dlio,
  title={DLIO: A Data-Centric Benchmark for Scientific Deep Learning Applications},
  author={H. Devarajan and H. Zheng and A. Kougkas and X.-H. Sun and V. Vishwanath},
  booktitle={IEEE/ACM International Symposium in Cluster, Cloud, and Internet Computing (CCGrid'21)},
  year={2021},
  volume={},
  number={81--91},
  pages={},
  publisher={IEEE/ACM}
}

@article{balmau2022mlperfstorage,
  title={Characterizing I/O in Machine Learning with MLPerf Storage},
  author={O. Balmau},
  booktitle={SIGMOD Record DBrainstorming},
  year={2022},
  volume={51},
  number={3},
  publisher={ACM}
}
```



## Acknowledgements
This work used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility under Contract DE-AC02-06CH11357 and is supported in part by National Science Foundation under NSF, OCI-1835764 and NSF, CSR-1814872.


## License
Apache 2.0 

