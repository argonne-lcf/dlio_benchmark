# Overview

This benchmark's objective is to simulate UNET3D, BERT and DLRM training workloads, and measure their I/O impact.

The training workloads were taken from the [MLCommons training benchmark repository](https://github.com/mlcommons/training).
We ran and characterized these workloads using [eBPF traces](https://github.com/iovisor/bpftrace) to understand their I/O behaviour, and attempted to reproduce them as closely as possible. You can find the traces we used, along with other stuff, in our [organization repository](https://github.com/discslab-dl-bench).

This tool is a fork of [DLIO](https://github.com/argonne-lcf/dlio_benchmark/), with the following features added:
- Dockerfile with dependencies
- support for evaluation phases on a held-out dataset
- support for PyTorch DataLoader reading NPZ files
- microsecond precision logging

Note: These additions have only been tested for TFRecord files, and PyTorch Dataloader with NPZ files as this is what our workload use. Because of this, other functionality is probably broken. More specifically, we're not trying to support the HDF5, CSV and NPZ readers, and we don't use Darshan or Tensorboard profiling.

# Supported Workloads

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

  <br>
  We are also looking to port each of these models to the other framework, in order to evaluate if there is a difference in I/O.

# Installation

## Pre-requisites
You must have docker installed on your system. Refer to https://docs.docker.com/get-docker/ for instructions.

## Installation Instructions

Clone the repository.
```
git clone <repository_address>/dlio_benchmark
cd dlio_benchmark/
```

Build the docker image.
```bash
sudo docker build -t dlio:<tag> .
```
The image name can be anything you want. The `<tag>` field is optionnal and serves to differentiate versions of an image.

<br>

# Running the benchmark

A DLIO run is split in 3 phases: 
- Generate synthetic data DLIO will use
- Run the benchmark using the previously generated data
- Post-process the results to generate a report

`start_dlio.sh` provides a convenient way to run these steps, by copying the given scripts to the container and running them within. This way, we don't have to rebuild the image evyer time we modify the scripts. The script will flush the caches on the host between data-generation and running the benchmark, as well as start `iostat` to gather device-level I/O information. 

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

Note: Make sure to remove or rename the data-directory between runs of different workloads or else the run will fail.

## UNET3D

To run the UNET3D simulation:
```
sudo ./start_dlio.sh -im <image:tag> -dgs workloads/UNET3D/datagen.sh -rs workloads/UNET3D/run.sh -pps workloads/UNET3D/postproc.sh -bd <dev-to-trace>
```

You can include multiple `-bd` options to trace multiple devices.

## BERT

To run the BERT simulation:
```
sudo ./start_dlio.sh -im <image:tag> -dgs workloads/BERT/datagen.sh -rs workloads/BERT/run.sh -pps workloads/BERT/postproc.sh -bd <dev-to-trace>
```
## DLRM
Work in progress.

<br>

## Command line options for DLIO

```
$ python3 src/dlio_benchmark.py -h
usage: dlio_benchmark.py [-h] [-fr {tensorflow,pytorch}] [-f {tfrecord,hdf5,csv,npz,hdf5_opt,data_loader}] [-r {off,seed,random}] [-ms SHUFFLE_SIZE] [-m {off,seed,random}] [-rt {memory,on_demand}] [-fa {multi,shared,collective}] [-rl RECORD_LENGTH] [-nf NUM_FILES_TRAIN] [-sf NUM_SAMPLES] [-bs BATCH_SIZE] [-e EPOCHS]
                         [-se SEED_CHANGE_EPOCH] [-gd GENERATE_DATA] [-go GENERATE_ONLY] [-df DATA_FOLDER] [-of OUTPUT_FOLDER] [-lf LOG_FILE] [-fp FILE_PREFIX] [-k KEEP_FILES] [-p PROFILING] [-l LOGDIR] [-s SEED] [-c DO_CHECKPOINT] [-cae CHECKPOINT_AFTER_EPOCH] [-ebc EPOCHS_BETWEEN_CHECKPOINTS] [-sbc STEPS_BETWEEN_CHECKPOINTS]
                         [-ts TRANSFER_SIZE] [-tr READ_THREADS] [-tc COMPUTATION_THREADS] [-ct COMPUTATION_TIME] [-rp PREFETCH] [-ps PREFETCH_SIZE] [-ec ENABLE_CHUNKING] [-cs CHUNK_SIZE] [-co {none,gzip,lzf,bz2,zip,xz}] [-cl COMPRESSION_LEVEL] [-d DEBUG] [-tts TOTAL_TRAINING_STEPS] [-de DO_EVAL] [-bse BATCH_SIZE_EVAL]
                         [-nfe NUM_FILES_EVAL] [-et EVAL_TIME] [-eae EVAL_AFTER_EPOCH] [-ebe EPOCHS_BETWEEN_EVALS] [-mos MODEL_SIZE]

DLIO Benchmark

optional arguments:
  -h, --help            show this help message and exit
  -fr {tensorflow,pytorch}, --framework {tensorflow,pytorch}
                        framework to use.
  -f {tfrecord,hdf5,csv,npz,hdf5_opt,data_loader}, --format {tfrecord,hdf5,csv,npz,hdf5_opt,data_loader}
                        data reader to use.
  -r {off,seed,random}, --read-shuffle {off,seed,random}
                        Shuffle the list of files to be read.
  -ms SHUFFLE_SIZE, --shuffle-size SHUFFLE_SIZE
                        (TF only) Size of the shuffle buffer in bytes.
  -m {off,seed,random}, --memory-shuffle {off,seed,random}
                        Shuffle the records returned by the data loader.
  -rt {memory,on_demand}, --read-type {memory,on_demand}
                        The read behavior for the benchmark.
  -fa {multi,shared,collective}, --file-access {multi,shared,collective}
                        How the files are accessed in the benchmark.
  -rl RECORD_LENGTH, --record-length RECORD_LENGTH
                        Size of a record/image within dataset
  -nf NUM_FILES_TRAIN, --num-files-train NUM_FILES_TRAIN
                        Number of files that should be accessed for training.
  -sf NUM_SAMPLES, --num-samples NUM_SAMPLES
                        Number of samples per file.
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Per worker batch size for training records.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to be emulated within benchmark.
  -se SEED_CHANGE_EPOCH, --seed-change-epoch SEED_CHANGE_EPOCH
                        change seed between epochs. y/n
  -gd GENERATE_DATA, --generate-data GENERATE_DATA
                        Enable generation of data. y/n
  -go GENERATE_ONLY, --generate-only GENERATE_ONLY
                        Only generate files then exit.
  -df DATA_FOLDER, --data-folder DATA_FOLDER
                        Set the path of folder where data is present in top-level.
  -of OUTPUT_FOLDER, --output-folder OUTPUT_FOLDER
                        Set the path of folder where output can be generated (checkpoint files and logs)
  -lf LOG_FILE, --log-file LOG_FILE
                        Name of the logfile
  -fp FILE_PREFIX, --file-prefix FILE_PREFIX
                        Prefix for generated files.
  -k KEEP_FILES, --keep-files KEEP_FILES
                        Keep files after benchmark. y/n
  -p PROFILING, --profiling PROFILING
                        Enable I/O profiling within benchmark. y/n
  -l LOGDIR, --logdir LOGDIR
                        Log Directory for profiling logs.
  -s SEED, --seed SEED  The seed to be used shuffling during read/memory.
  -c DO_CHECKPOINT, --do-checkpoint DO_CHECKPOINT
                        Enable checkpointing. y/n
  -cae CHECKPOINT_AFTER_EPOCH, --checkpoint-after-epoch CHECKPOINT_AFTER_EPOCH
                        Epoch number after which to enable checkpointing.
  -ebc EPOCHS_BETWEEN_CHECKPOINTS, --epochs-between-checkpoints EPOCHS_BETWEEN_CHECKPOINTS
                        Number of epochs between checkpoints.
  -sbc STEPS_BETWEEN_CHECKPOINTS, --steps-between-checkpoints STEPS_BETWEEN_CHECKPOINTS
                        Number of steps between checkpoints.
  -ts TRANSFER_SIZE, --transfer-size TRANSFER_SIZE
                        Transfer Size for tensorflow buffer size.
  -tr READ_THREADS, --read-threads READ_THREADS
                        Number of threads to be used for reads.
  -tc COMPUTATION_THREADS, --computation-threads COMPUTATION_THREADS
                        Number of threads to be used for pre-processing.
  -ct COMPUTATION_TIME, --computation-time COMPUTATION_TIME
                        Processing time (seconds) for each training data batch.
  -rp PREFETCH, --prefetch PREFETCH
                        Enable prefetch within benchmark.
  -ps PREFETCH_SIZE, --prefetch-size PREFETCH_SIZE
                        Number of batches to prefetch.
  -ec ENABLE_CHUNKING, --enable-chunking ENABLE_CHUNKING
                        Enable chunking for HDF5 files.
  -cs CHUNK_SIZE, --chunk-size CHUNK_SIZE
                        Set chunk size in bytes for HDF5.
  -co {none,gzip,lzf,bz2,zip,xz}, --compression {none,gzip,lzf,bz2,zip,xz}
                        Compression to use.
  -cl COMPRESSION_LEVEL, --compression-level COMPRESSION_LEVEL
                        Level of compression for GZip.
  -d DEBUG, --debug DEBUG
                        Enable debug in code.
  -tts TOTAL_TRAINING_STEPS, --total-training-steps TOTAL_TRAINING_STEPS
                        Total number of training steps to take. DLIO will terminate after this number.
  -de DO_EVAL, --do-eval DO_EVAL
                        If we should simulate evaluation (single rank only for now). See -et, -eae and -eee to configure.
  -bse BATCH_SIZE_EVAL, --batch-size-eval BATCH_SIZE_EVAL
                        Per worker batch size for evaluation records.
  -nfe NUM_FILES_EVAL, --num-files-eval NUM_FILES_EVAL
                        Number of files that should be put aside for evaluation. Defaults to zero.
  -et EVAL_TIME, --eval-time EVAL_TIME
                        Processing time (seconds) for each evaluation data batch.
  -eae EVAL_AFTER_EPOCH, --eval-after-epoch EVAL_AFTER_EPOCH
                        Epoch number after which to start evaluating
  -ebe EPOCHS_BETWEEN_EVALS, --epochs-between-evals EPOCHS_BETWEEN_EVALS
                        Evaluation frequency: evaluate every x epochs
  -mos MODEL_SIZE, --model-size MODEL_SIZE
                        Size of the model (for checkpointing) in bytes
```

# Benchmark Submission

TBD

# Current Limitations and Future Work

There is an abtraction problem where DLIO considers PyTorch's data loader as a data type when it really is data type agnostic.
Any of the existing data formats could be supported by extending the Dataset class and implementing the [\_\_len__() and \_\_getitem__() methods](https://pytorch.org/docs/stable/data.html). This forces us to e.g. use `npz_generator` when generating, but switch to `data_loader_reader` when running, which is confusing.


DLIO currently assumes the samples to always be 2D images, which is the case for none of our workloads. UNET3D operates on 3D images, which we faked by pretending to have multiple samples in a file (effectively the 3rd dimension). In reality, UNET3D reads in 2 files in every iteration (one for the sample and one for the label), which we can't imitate presently.

Similarly, it hardcodes the TFRecord format to be image/label pairs in the generator and reader, when the records for BERT do not follow this format at all. We will have to modify this for DLRM which reads in a custom binary format.

Considering this and our objectives, it might make sense to change the DLIO abstraction to be workload based, instead of framework/datareader based.
Under a `workload/` folder, we would add BERT, DLRM, UNET3D and could implement their simulations more precisely than by trying to bend DLIO into reproducing them when it was not built for this purpose.

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

