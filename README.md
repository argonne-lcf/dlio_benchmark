# Deep Learning I/O (DLIO) Benchmark
![test status](https://github.com/argonne-lcf/dlio_benchmark/actions/workflows/python-package-conda.yml/badge.svg)

This README provides a abbreviated documentation of the DLIO code. Please refer to https://argonne-lcf.github.io/dlio_benchmark/ for full user documentation. 

## Overview

DLIO is an I/O benchmark for Deep Learning. DLIO is aimed at emulating the I/O behavior of various deep learning applications. The benchmark is delivered as an executable that can be configured for various I/O patterns. It uses a modular design to incorporate more data loaders, data formats, datasets, and configuration parameters. It emulates modern deep learning applications using Benchmark Runner, Data Generator, Format Handler, and I/O Profiler modules. 

## Installation and running DLIO
### Bare metal installation 

```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
pip install -r requirements.txt
export PYTHONPATH=$PWD/:$PYTHONPATH
python ./src/dlio_benchmark.py ++workload.workflow.generate_data=True

Additionally, to generate the report `iostat` is needed and can be installed from the `sysstat` package using your package manager.

```
## Container

```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
docker build -t dlio .
docker run -t dlio python ./src/dlio_benchmark.py ++workload.workflow.generate_data=True
``` 

You can also pull rebuilt container from docker hub: 
```bash
docker docker.io/zhenghh04/dlio:latest
docker run -t docker.io/zhenghh04/dlio:latest python ./src/dlio_benchmark.py ++workload.workflow.generate_data=True
```

One can also run interactively 
```bash
docker run -t docker.io/zhenghh04/dlio:latest /bin/bash
root@30358dd47935:/workspace/dlio# python ./src/dlio_benchmark.py ++workload.workflow.generate_data=True
```


## Running the benchmark

A DLIO run is split in 3 phases: 
- Generate synthetic data DLIO will use
- Run the benchmark using the previously generated data
- Post-process the results to generate a report

The configurations of a workload can be specified through a yaml file. Examples of yaml files can be found in [./configs/workload/](./configs/workload). 

One can specify the workload through the ```workload=``` option on the command line. Specific configuration fields can then be overridden following the ```hydra``` framework convention (e.g. ```++workload.framework=tensorflow```). 

First, generate the data
  ```bash
  mpirun -np 8 python3 src/dlio_benchmark.py workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False
  ```
If possible, one can flush the filesystem caches in order to properly capture device I/O
  ```bash
  sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
  ```
Finally, run the benchmark with ```iostat``` profiling, listing the io devices you would like to trace.
  ```bash
  mpirun -np 8 python3 src/dlio_benchmark.py workload=unet3d ++workload.workflow.profiling=True ++workload.profiling.profiler=iostat ++workload.profiling.iostat_devices=[sda,sdb]
  ```

All the outputs will be stored in ```hydra_log/unet3d/$DATE-$TIME``` folder. To post process the data, one can do
```bash 
python3 src/dlio_postprocessor.py --output-folder hydra_log/unet3d/$DATE-$TIME
```
This will generate ```DLIO_$model_report.txt``` in the output folder. 

## Workload YAML configuration file 
Workload characteristics are specified by a YAML configuration file. Below is an example of a YAML file for the UNet3D workload which is used for 3D image segmentation. 

```
  # contents of unet3d.yaml
  model: unet3d

  framework: pytorch

  workflow:
    generate_data: False
    train: True
    evaluation: True

  dataset: 
    data_folder: ./data/unet3d/
    format: npz
    num_files_train: 3620
    num_files_eval: 42
    num_samples_per_file: 1
    batch_size: 4
    batch_size_eval: 1
    file_access: multi
    record_length: 1145359
    keep_files: True
  
  data_reader: 
    data_loader: pytorch
    read_threads: 4
    prefetch: True

  train:
    epochs: 10
    computation_time: 4.59

  evaluation: 
    eval_time: 11.572
    epochs_between_evals: 2
```

The full list of configurations can be found in: https://argonne-lcf.github.io/dlio_benchmark/config.html

The YAML file is loaded through hydra (https://hydra.cc/). The default setting are overridden by the configurations loaded from the YAML file. One can override the configuration through command line (https://hydra.cc/docs/advanced/override_grammar/basic/). 

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
```

We also encourage people to take a look at a relevant work from MLPerf Storage working group. 

```
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

## Acknowledgments

This work used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility under Contract DE-AC02-06CH11357 and is supported in part by National Science Foundation under NSF, OCI-1835764 and NSF, CSR-1814872.

## License

Apache 2.0 [LICENSE](./LICENSE)

---------------------------------------
Copyright © 2022, UChicago Argonne, LLC
All Rights Reserved

If you have questions about your rights to use or distribute this software, please contact Argonne Intellectual Property Office at partners@anl.gov

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
