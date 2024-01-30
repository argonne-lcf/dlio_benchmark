.. _yaml: 

DLIO Configuration
==============================================
The characteristics of a workload is specified through a YAML file. This file will then be read by `DLIO` to setup the benchmark. Below is an example of such a YAML file. 

.. code-block:: yaml
  
  model: unet3d

  framework: pytorch

  workflow:
    generate_data: False
    train: True
    checkpoint: True

  dataset: 
    data_folder: data/unet3d/
    format: npz
    num_files_train: 168
    num_samples_per_file: 1
    record_length: 146600628
    record_length_stdev: 68341808
    record_length_resize: 2097152
    
  reader: 
    data_loader: pytorch
    batch_size: 4
    read_threads: 4
    file_shuffle: seed
    sample_shuffle: seed

  train:
    epochs: 5
    computation_time: 1.3604

  checkpoint:
    checkpoint_folder: checkpoints/unet3d
    checkpoint_after_epoch: 5
    epochs_between_checkpoints: 2
    model_size: 499153191

More examples can be found in the `workload`_ folder. One can also create custom configuration file. How to load custom configuration file can be found in :ref:`run`. 

A `DLIO` YAML configuration file contains following sections: 

* **model** - specifying the name of the model.
* **framework** - specifying the framework to use for the benchmark, options: tensorflow, pytorch
* **workflow** - specifying what workflow operations to perform. Workflow operations include: dataset generation (``generate_data``), training (``train``), evaluation (``evaluation``), checkpointing (``checkpoint``), debugging (``debug``), etc. 
* **dataset** - specifying all the information related to the dataset. 
* **reader** - specifying the data loading options 
* **train** - specifying the setup for training
* **evaluation** - specifying the setup for evaluation. 
* **checkpoint** - specifying the setup for checkpointing. 
* **profiling** - specifying the setup for profiling

model
------------------
One can specify the name of the model as 

.. code-block:: yaml

  model: unet3d

No other parameters under this section. 


framework
-------------------
Specify the frameork (tensorflow or pytorch) as 

.. code-block:: yaml

  framework: tensorflow

No parameters under this group. 


workflow
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - generate_data
     - False
     - whether to generate dataset
   * - train
     - True
     - whether to perform training
   * - evaluation
     - False
     - whether to perform evaluation
   * - checkpoint
     - False
     - whether to perform checkpointing
   * - profiling
     - False
     - whether to perform profiling

.. note:: 

  If ``train`` is set to be ```False```, ``evaluation``, ``checkpoint``, ``profiling`` will be set to ```False``` automatically. 

  Even though ``generate_data`` and ``train`` can be performed together in one job, we suggest to perform them seperately. One can generate the data first by running DLIO with ```generate_data=True``` and ```train=False```, and then run training benchmark with ```generate_data=False``` and ```train=True```. 

dataset
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - record_length
     - 65536
     - size of each sample
   * - record_length_stdev
     - 0.
     - standard deviation of the size of samples
   * - record_length_resize
     - 0. 
     - resized sample size 
   * - format
     - tfrecord
     - data format [tfrecord|csv|npz|jpeg|png]
   * - num_files_train
     - 1
     - number of files for the training set
   * - num_files_eval
     - 0
     - number of files for evaluation/validation set
   * - num_samples_per_file
     - 1
     - number of samples per file
   * - data_folder
     - ./data
     - the path to store the dataset. 
   * - num_subfolders_train
     - 0
     - number of subfolders that the training set is stored
   * - num_subfolders_eval
     - 0
     - number of subfolders that the evaluation/validation set is stored
   * - file_prefix
     - img
     - the prefix of the dataset file(s)
   * - compression
     - none
     - what compressor to use to compress the dataset. (limited support)
   * - compression_level
     - 4
     - level of compression for gzip
   * - chunking
     - False
     - whether to use chunking to store hdf5. 
   * - chunk_size
     - 0
     - the chunk size for hdf5. 
   * - keep_files
     - True
     - whether to keep the dataset files afer the simulation.    

.. note :: 
  The training and validation datasets will be put in ```${data_folder}/train``` and ```${data_folder}/valid``` respectively. If ``num_subfolders_train`` and ``num_subfolders_eval`` are larger than one, the datasets will be split into multiple subfolders within ```${data_folder}/train``` and ```${data_folder}/valid```. 

.. note :: 
  The DALI data loader configuration needs to be coupled with pytorch framework otherwise will fail.

.. attention::
  
  For `format: jpeg`, it is not recommended to generate data due to its lossy compression nature. Instead, provide the path to original dataset in the `data_folder` parameter. 
  More information on JPEG image generator analysis is provided at :ref:`jpeg_generator_issue` section. 
  Follow the original dataset directory structure as described in :ref:`directory structure <directory-structure-label>`
  
reader 
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - data_loader
     - tensorflow
     - select the data loader to use [tensorflow|pytorch|dali|native_dali]. 
   * - batch_size
     - 1 
     - batch size for training
   * - batch_size_eval
     - 1 
     - batch size for evaluation
   * - read_threads* 
     - 1
     - number of threads to load the data (for tensorflow and pytorch data loader)
   * - computation_threads
     - 1
     - number of threads to preprocess the data
   * - prefetch_size
     - 0
     - number of batches to prefetch (0 - no prefetch at all)
   * - sample_shuffle
     - off
     - [seed|random|off] whether and how to shuffle the dataset samples
   * - file_shuffle
     - off
     - [seed|random|off] whether and how to shuffle the dataset file list
   * - transfer_size
     - 262144
     - transfer size in byte for tensorflow data loader. 
   * - preprocess_time
     - 0.0
     - The amount of emulated preprocess time (sleep) in second. 
   * - preprocess_time_stdev
     - 0.0
     - The standard deviation of the amount of emulated preprocess time (sleep) in second. 
.. note:: 

  TensorFlow and PyTorch behave differently for some parameters. For ``read_threads``, tensorflow does 
  not support ``read_threads=0``, but pytorch does, in which case, the main thread will be doing data loader and no overlap between I/O and compute. 

  For pytorch, ``prefetch_size`` is set to be 0, it will be changed to 2. In other words, the default value for ``prefetch_size`` in pytorch is 2. 

  For Dali data loader, we support two options, ``dali`` and ``native_dali```. ``dali`` uses our internal reader, such as ``jpeg_reader``, ``hdf5_reader``, etc, and ``dali.fn.external_source``; whereas ``native_dali`` directly uses Dali readers, such as ``dn.readers.numpy``, ``fn.readers.tfrecord``, and ``fn.readers.file``. 


train
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - epochs
     - 1
     - number of epochs to simulate
   * - computation_time
     - 0.0
     - emulated computation time per step in second
   * - computation_time_stdev
     - 0.0
     - standard deviation of the emulated computation time per step in second
   * - total_training_steps
     - -1
     - number of training steps to simulate, assuming running the benchmark less than one epoch. 
   * - seed_change_epoch
     - True
     - whether to change random seed after each epoch
   * - seed
     - 123
     - the random seed     

evaluation
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - eval_time
     - 0
     - emulated computation time (sleep) for each evaluation step. 
   * - eval_time_stdev
     - 0
     - standard deviation of the emulated computation time (sleep) for each evaluation step. 
   * - epochs_between_evals
     - 1
     - evaluate after x number of epochs

checkpoint
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - checkpoint_folder
     - ./checkpoints/
     - the folder to save the checkpoints
   * - checkpoing_after_epoch
     - 1
     - start checkpointing after certain number of epochs specified 
   * - epochs_between_checkpoints
     - 1
     - performing one checkpointing per certain number of epochs specified
   * - steps_between_checkpoints
     - -1
     - performing one checkpointing per certain number of steps specified
   * - model_size
     - 10240
     - the size of the model parameters per GPU in bytes
   * - optimization_groups
     - []
     - List of optimization group tensors. Use Array notation for yaml.
   * - num_layers
     - 1
     - Number of layers to checkpoint. Each layer would be checkpointed separately.
   * - layer_parameters
     - []
     - List of parameters per layer. This is used to perform I/O per layer.
   * - type
     - rank_zero
     - Which rank performs this checkpoint. All ranks (all_ranks) or Rank 0 (rank_zero).
   * - tensor_parallelism
     - 1
     - Tensor parallelism for model. Used to determine the number of layer model files.
   * - pipeline_parallelism
     - 1
     - Pipeline parallelism for model.

.. note::
   
   By default, if checkpoint is enabled, it will perform checkpointing from every epoch.

   One can perform multiple checkpoints within a single epoch, by setting ``steps_between_checkpoints``. If ``steps_between_checkpoints`` is set to be a positive number, ``epochs_between_checkpoints`` will be ignored.
   

output
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - folder
     - None
     - The output folder name.
   * - log_file
     - dlio.log
     - log file name  

.. note::
   
   If ``folder`` is not set (None), the output folder will be ```hydra_log/unet3d/$DATE-$TIME```. 

profiling
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - iostat_devices**
     - [sda, sdb]
     - specifying the devices to perform iostat tracing.  

.. note::
   
   We support multi-level profiling using:
    * ``dlio_profiler``: https://github.com/hariharan-devarajan/dlio-profiler. DLIO_PROFILER_ENABLE=1 has to be set to enable profiler.

The YAML files are stored in the `workload`_ folder.
It then can be loaded by ```dlio_benchmark``` through hydra (https://hydra.cc/). This will override the default settings. One can override the configurations through command line (https://hydra.cc/docs/advanced/override_grammar/basic/).


.. _workload: https://github.com/argonne-lcf/dlio_benchmark/tree/main/dlio_benchmark/configs/workload
