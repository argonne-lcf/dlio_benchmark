.. _yaml: 

Fully Reference of YAML Configuration 
==============================================
The configure file contains following groups: 

* **model** - specifying the name of the model.
* **framework** - specifying the framework to use for the benchmark, options: tensorflow, pytorch
* **workflow** - specifying what workflow operations to perform, including dataset generation, training, evaluation, checkpointing, evaluation, debugging, etc. 
* **dataset** - specifying all the information related to the dataset. 
* **data_reader** - specifying the data loading options 
* **train** - specifying the setup for training
* **evaluation** - specifying the setup for evaluation. 
* **checkpoint** - specifying the setup for checkpointing. 

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
     - none
     - do profiling and specify the profiler

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
     - the path to store the dataset
   * - num_subfolders_train
     - 0
     - number of subfolders that the training set is stored
   * - num_subfolders_eval
     - 0
     - number of subfolders that the evaluation/validation set is stored
   * - batch_size
     - 1 
     - batch size for training
   * - batch_size_eval
     - 1 
     - batch size for evaluation
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

data_reader 
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - data_loader
     - tensorflow
     - select the data loader to use [tensorflow|pytorch|node]  
   * - read_threads
     - 1
     - number of threads to load the data (for tensorflow and pytorch data loader)
   * - computation_threads
     - 1
     - number of threads to preprocess the data
   * - prefetch
     - False
     - whether to prefetch the dataset
   * - prefetch_size
     - 0
     - number of batch to prefetch
   * - read_shuffle
     - off
     - [seed|random|off] whether and how to shuffle the dataset
   * - file_access
     - multi
     - multi - file per process; shared - independent access to a single shared file; collective - collective I/O access to a single shared file
   * - transfer_size
     - 1048576
     - transfer size in byte for tensorflow data loader. 


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
   * - epochs_between_evals
     - 0
     - evaluate after x number of epochs

checkpoint
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - checkpoing_after_epoch
     - 0
     - start checkpointing after certain number of epochs specified 
   * - epochs_between_checkpoints
     - 0
     - performing one checkpointing per certain number of epochs specified
   * - steps_between_checkpoints
     - 0
     - performing one checkpointing per certain number of steps specified
   * - model_size
     - 10240
     - the size of the model in bytes
