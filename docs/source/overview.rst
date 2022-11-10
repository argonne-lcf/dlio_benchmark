Introduction
=============
Deep learning has been shown as a successful
method for various tasks, and its popularity results in numerous
open-source deep learning software tools. Deep learning has
been applied to a broad spectrum of scientific domains such
as cosmology, particle physics, computer vision, fusion, and
astrophysics. Scientists have performed a great deal of work
to optimize the computational performance of deep learning
frameworks. However, the same cannot be said for I/O performance. As deep learning algorithms rely on big-data volume and
variety to effectively train neural networks accurately, I/O is
a significant bottleneck on large-scale distributed deep learning training. 

The DLIO benchmark aims to provide a detailed representation of
the I/O behavior of deep learning workloads. DLIO can be utilized to accurately emulate the I/O behavior of modern deep learning
applications. Using DLIO, application developers and system
software solution architects can identify potential I/O bottlenecks
in their applications and guide optimizations to boost the I/O
performance. The storage vendors can also use DLIO benchmark as a guidance for storage and file system design. 

Foundation of DLIO benchmark 
==========================================
First, we assume that one can replace the computation part (training and validation) with a sleep of the same amount of time, while keeping the I/O pattern / behavior the same. The logic behind this is demonstrated as shown in the figure. In a typical deep leanring training process, a batch of data is loaded from the storage to host memory at each time step, and then transfered to the accelerator to perform the training. There might be some hardware supporting loading data from storage directly to the accelerators such as GPU Direct. In either case, the I/O (data access in the storage) should be independent of what is going on inside the accelerator, as long as the frequency of the I/O requests remains the same. 

  .. figure:: ./images/training.png

    Typical process of AI training. The dataset is loaded from the storage to the host RAM and then feed into the accelerators for training. The storage benchmarks will focus on data loading from the storage to the host RAM. 

We have validated this in various cases. For example, in the figure shown below, we replace the computation with a sleep of different amounts corresponding to the training time in Nvidia A100, V100, and P100 GPUs, we were able to reproduce the I/O timeline trace of the real workload running on different GPUs. More results from distributed training were presented in our CCGrid paper. 

  .. figure:: ./images/validation.png

    Upper panel: I/O timeline on A100, V100, P100; Lower panel: I/O timeline on Skylake with training replaced by sleep of different amounts of time equal to the training time on A100, V100 and P100 respectively. 

Second, one can have certain extent of abstraction of the dataset and igore the low level details. We assume that as long as the number of files, number of samples per file, size of each sample, batch size, are the same, the I/O behavior should be similar regardless of the details of each sample. We incorporate built-in data loaders such as tf.data, and torch DataLoader to incorporate advance features such as prefetch, and multithreaded data loading. 

High-level Design
=======================
The benchmark uses a modular design to incorporate
more data formats, datasets, and configuration parameters. It
emulates deep learning applications using
**Benchmark Runner**, **Data Generator**, **Format Handler**, and **I/O Profiler** modules. These modules utilize state-of-the-art design
patterns to build a transparent and extensible framework. The
DLIO benchmark has been designed with the following goals.

1) Accurate: DLIO should be an accurate representation of
selected deep learning applications. It should
incorporate all the I/O behavior seen in various configurations of applications, and act as a mini-application that can precisely replay the I/O behavior. 

2) Configurable: DLIO should be easily configurable for
different scenarios required by the user. These include
features such as different ratio-of-computation to I/O, multi
threading for I/O, data operators (e.g., decoding, shuffling,
prefetch, and batching), and mechanism to feed data into training.

3) Extensible: DLIO benchmark should allow adding
custom data directories and enable easy extensions to the
benchmark to incorporate different data formats, data loaders or data generation algorithms. These changes should not affect
the basic benchmark operations.

''''''''''''''''''''
DLIO code modules
''''''''''''''''''''
Below shows the modules of the DLIO code. 

.. image:: images/dlio.png

* **Configuration Manager**: the user specifies a YAML file which represents the characteristics of a real workload. The configuration manager will load the configuration into DLIO. 

* **Format Handler**: Format Handler will handle the data read and write for specific data format. 

* **Data Generator**: this is for generating synthetic datasets. 

* **Benchmark Runner**: this is for performing the whole benchmarking process, including data generation, training, evaluation, checkpointing, profiling, etc. 

'''''''''''''''''''''''''
Workload Configuration
'''''''''''''''''''''''''
The characteristics of a workload is specified through a set of configuration in a YAML file. Below is an example of a YAML file for UNet3D workload which was used for 3D image segmentation. 

.. block:: 
  
  model: unet3d

  framework: pytorch

  workflow:
    generate_data: True
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

The full list of configurations can be found in: :ref:`yaml`. 

The YAML file is loaded through hydra (https://hydra.cc/). The default setting are overridden by the configurations loaded from the YAML file. One can override the configuration through command line (https://hydra.cc/docs/advanced/override_grammar/basic/). 


.. code-block:: bash

  $ python src/dlio_benchmark.py --help

'''''''''''''''''''''
Dataset Generation
'''''''''''''''''''''
Data Generator can generate synthetic datasets based on the configuration specified in the YAML file. This eliminates the dependence on real dataset which is typically difficult to get. One can also scale the dataset to study the I/O load at different scales. The dataset generation process can be done in parallel. 

The organization, layout and storage of the data on the file system can have significant impact on the data and metadata load and store. We support different data organizations, such as 

* Single shared file in which the entire datasets is stored in one file. 
* One samples per file
* Multiple samples per file
* Files putting in a single folder. 
* Files putting in many subfolders.  

All the options are configurable in the "dataset" section of the YAML file. 

'''''''''''''''''''''''
Profiling
'''''''''''''''''''''''
In the profiling module, we support following I/O profiling using following profilers: 

* Darshan: https://www.mcs.anl.gov/research/projects/darshan/

* iostat: https://linux.die.net/man/1/iostat

* tf.profiler: https://www.tensorflow.org/api_docs/python/tf/profiler

* torch.profiler: https://pytorch.org/docs/stable/profiler.html

This can be enabled through through the "workflow" section in the workload YAML file. 

'''''''''''''''''''''''
DLIO excution
'''''''''''''''''''''''
**Configuration**: The YAML file is first parsed and extracted into configurations for the benchmark. The extracted configurations are passed to the Configuration Manager, which is first initialized with default benchmark values and then updates itself with the incoming configurations. At this stage, incompatible/incorrect configurations would be thrown as error back to the users. 

**Data generation**: Once the configurations are validated and applied, the benchmark runner is invoked. The runner initializes prepared data (if needed) and then starts the profiling session. 

**Simulation**: Once the session has started successfully, the benchmark Run() is invoked, which runs the benchmark. In the run phase, we run the benchmark for multiple epochs. During each epoch, the whole data is read once using n steps. During an epoch, checkpoint operations are performed every c steps as well. 

Additionally, an inter-step computation is performed to emulate computation (through a sleep function) and I/O phases by deep learning application. Replacing computaiton with sleep allows the user to perform the benchmark in a acclerator absence environement. Different accelerators will have different amounts of computation time. 

Finally, once the benchmark run finishes, the finalize is called, which stops the profiler, saves its results, and exits the benchmark.

**Post processing**: One can then use the post processing script to process the logs to produce a high level summary of the I/O performance. 