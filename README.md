# Deep Learining I/O (DLIO) Benchmark

Documentation: https://argonne-lcf.github.io/dlio_benchmark/

## Overview

DLIO is an I/O benchmark for Deep Learning. DLIO is aimed at emulating the I/O behavior of various deep learning applications. The benchmark is delivered as an executable that can be configured for various I/O patterns. It uses a modular design to incorporate more data loaders, data formats, datasets, and configuration parameters. It emulates modern deep learning applications using Benchmark Runner, Data Generator, Format Handler, and I/O Profiler modules. 

### Features 
* Easy-to-use and highly configurable argument list to emulate deep learning application's I/O behavior.
* Able to generate synthetic datasets for different deep learning applications. 
* Full transparency over emulation of I/O access with logging at different levels.
* Easy to use data generator to test the performance of different data layouts and its impact on the I/O performance.
* Compatible with modern profiling tools such as Tensorboard and Darshan to extract and analyze I/O behavior.

## Installation and running DLIO

```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
pip install -r requirements.txt 
python ./src/dlio_benchmark.py --help
```

## Running the benchmark

A DLIO run is split in 3 phases: 
- Generate synthetic data DLIO will use
- Run the benchmark using the previously generated data
- Post-process the results to generate a report

The configurations of a workload can be specified through a yaml file. Examples of yaml files can be find in [./configs/workload/](./configs/workload). 

One can specify specify workload through ```workload=``` option in the command line. The configuration can be overridden through commandline in the ```hyra``` framework (e.g.```++workload.framework=tensorflow```). 

For example, to run the unet3d benchark, one can do
```bash
mpirun -np 8 python src/dlio_benchmark.py workload=unet3d
```
This will both generate the dataset and perform benchmark. 

One can separate the data generation part and training part as 
* data generation
  ```bash
  mpirun -np 8 python src/dlio_benchmark.py workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False ++workload.workflow.evaluation=False
  ```
* running benchmark
  ```bash
  mpirun -np 8 python src/dlio_benchmark.py workload=unet3d ++workload.workflow.generate_data=False ++workload.workflow.train=True ++workload.workflow.evaluation=True
  ```

All the outputs will be stored in ```hydra_log/unet3d/$DATE-$TIME``` folder. To post process the data, one can do
```bash 
python3 src/dlio_postprocessor.py --output_folder=hydra_log/unet3d/$DATE-$TIME
```
This will generate ```DLIO_$model_report.txt``` inside the output folder. 

## Workload YAML configuration file 

```
$ python3 src/dlio_benchmark.py --help
  The configuration can be specified by a yaml config file.

  A complete list of config options are: 

  workflow:
    generate_data: whether to generate data
    train: whether to perform training 
    debug: whether to turn on debugging
    profiling: profiler to be used. Default: none
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
We are open to the contribution from the community for the development of the benchmark. Specifically, we welcome contribution in the following aspects:
General new features needed including: 

* support for new workloads: if you think that your workload(s) would be interested to the public, and would like to provide the yaml file to be included in the repo, please submit an issue.  
* support for new data loaders, such as DALI loader, MxNet loader, etc
* support for new frameworks, such as MxNet
* support for noval file systems or storage, such as AWS S3. 
* support for loading new data formats. 

If you would like to contribute, please submit issue to https://github.com/argonne-lcf/dlio_benchmark/issues, and contact ALCF DLIO team, Huihuo Zheng at huihuo.zheng@anl.gov

## Citation and Reference
The original paper describe the design and implementation of DLIO code is as follows: 
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

We also encourage people to take a look at a relevant work from MLPerf Storage working group. 
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

Copyright@2021 UChicago Argonne LLC

