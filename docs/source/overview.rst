Introduction
=============
Deep learning has been shown as a successful
method for various tasks, and its popularity results in numerous
open-source deep learning software tools. It has
been applied to a broad spectrum of scientific domains such
as cosmology, particle physics, computer vision, fusion, and
astrophysics. As deep learning algorithms rely on big-data volume and
variety to effectively train neural networks accurately, I/O is
a significant bottleneck on large-scale distributed deep learning training. 

The `DLIO` benchmark aims to provide a detailed representation of
the data access pattern of deep learning workloads, to 
to accurately emulate the I/O behavior in the training process. 
Using `DLIO`, application developers and system
software solution architects can identify potential I/O bottlenecks
in their applications and guide optimizations to boost the I/O
performance. Storage hardware vendors can also use `DLIO` benchmark as 
a guidance for designing storage and file system 
targeting the deep learning applications. 

In developing the benchmark, we have the following two assumptions: 

First, we assume that one can replace the computation part 
(either training or validation) with a sleep function of the same amount of time, 
while keeping the I/O pattern / behavior the same. 
The logic behind this is demonstrated in the figure. 
In a typical deep learning training process, a batch of data is 
loaded from the storage either to host memory and then transfered 
to the accelerator or directly to
 the accelerator through networks such as GPU Direct. In either case, 
the I/O (data access in the storage) should likely be independent of 
what is going on inside the accelerator, whether it is doing busy computation or being idle. As long as amount of time per step is the same, the 
frequency of the I/O requests issued will be the same, which eventually results in the same I/O behavior on the storage.

  .. figure:: ./images/training.png

    Typical process of AI training. The dataset is loaded from the storage to the host RAM and then feed into the accelerators for training. The storage benchmarks will focus on data loading from the storage to the host RAM. 

We validated this assumption with UNet3D application. In the figure shown below, we replace the computation with a sleep of different amounts corresponding to the training time in Nvidia A100, V100, and P100 GPUs, we were able to reproduce the I/O timeline trace of the real workload running on different GPUs. 

  .. figure:: ./images/validation.png

    Upper panel: I/O timeline on A100, V100, P100; Lower panel: I/O timeline on Skylake with training replaced by sleep of different amounts of time equal to the training time on A100, V100 and P100 respectively. 

Replacing the training part with a sleep function eliminate the needs of actual accelerators to perform the I/O benchmark, which significantly reduces the cost and complexity of benchmarking. Not only so, it allows us to simulate the I/O pattern for different types of accelerators easily by simply changing the sleep time accordingly.

Second, one can have certain extent of abstraction and generalization on the data loading process. For the dataset, we assume that as long as the number of files, number of samples per file, size of each sample, batch size, and format are the same, the I/O behavior should be similar regardless of the details of each sample. If this assumption is true, one does not need to use the actual dataset to perform the benchmark. We can generate the dataset within `DLIO`. As long as the characteristics of the dataset remain the same, we are guranteed to have the same I/O behavior. On the other hand, to keep the data loading as close as the actual process, we incorporate built-in framework data loaders such as tf.data, torch DataLoader, and Dali data loader, so that `DLIO` can simulate the advance optimization features such as pipeline, prefetching, multithreaded data loading, etc. This makes DLIO able to represent to a broad spectrum of deep learning applications. 

High-level Design
=======================
The benchmark uses a modular design artitecture, which consists of
**Benchmark Runner**, **Data Generator**, **Format Handler**, and **I/O Profiler** modules. These modules utilize state-of-the-art design
patterns to build a transparent and extensible framework. It has been designed with the following goals.

1) **Accurate**: `DLIO` should be an accurate representation of
selected deep learning applications. It should
incorporate all the I/O behavior seen in various configurations of applications,
 and act as a mini-application that can precisely replay the I/O behavior. 

2) **Configurable**: `DLIO` should be easily configurable for
different scenarios required by the user. These include
features such as different ratio-of-computation to I/O, multi
threading for I/O, data operators (e.g., decoding, shuffling,
prefetch, and batching), and mechanism to feed data into training.

3) **Extensible**: `DLIO` benchmark should allow adding
custom data directories and enable easy extensions to the
benchmark to incorporate different data formats, data loaders or data generation algorithms. 
These changes should not affect the basic benchmark operations.

''''''''''''''''''''
`DLIO` Code Modules
''''''''''''''''''''
Below shows the modules of the `DLIO` code. 

.. image:: images/dlio.png

* **Configuration Manager**: the user specifies a YAML file which represents the characteristics of a real workload. The configuration manager will load the configuration into `DLIO`. 

* **Format Handler**: Format Handler will handle the data read and write for specific data format. 

* **Data Generator**: this is for generating synthetic datasets. This eliminates the dependence on real dataset which is typically difficult to get. `DLIO` can generate synthetic data in different formats, different organization and layouts on the storage, such as: 

  * Single shared file in which the entire datasets is stored in one file. 
  * One samples per file
  * Multiple samples per file
  * Files putting in a single folder. 
  * Files putting in many subfolders.  

* **Benchmark Runner**: this is for performing the whole benchmarking process, including data generation, training, evaluation, checkpointing, profiling, etc. 

'''''''''''''''''''''''
Benchmark Execution
'''''''''''''''''''''''
**Configuration**: The YAML configure file is first parsed and extracted into configurations for the benchmark. The extracted configurations are passed to the Configuration Manager, which is first initialized with default benchmark values and then updates itself with the incoming configurations. At this stage, incompatible/incorrect configurations would be thrown as error back to the users. A complete instruction on how to prepare the YAML file can be found in :ref:`yaml`. 

**Data generation**: Once the configurations are validated and applied, the benchmark runner is invoked. The runner initializes prepared data (if needed) and then starts the profiling session. 

**Simulation**: Once the session has started successfully, the benchmark Run() is invoked, which runs the benchmark. In the run phase, we run the benchmark for multiple epochs. During each epoch, the whole data is read once using n steps. During an epoch, checkpoint operations are performed every c steps as well. 

Additionally, an inter-step computation is performed to emulate computation (through a sleep function) and I/O phases by deep learning application. Replacing computaiton with sleep allows the user to perform the benchmark in a acclerator absence environement. Different accelerators will have different amounts of computation time. 

Finally, once the benchmark run finishes, the finalize is called, which stops the profiler, saves its results, and exits the benchmark.

**Post processing**: One can then use the post processing script to process the logs to produce a high level summary of the I/O performance. 

