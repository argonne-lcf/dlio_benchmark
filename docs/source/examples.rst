Examples
=============

We here list a set of example workloads. In the first example, we show the benchmarking process, including generating the dataset, running the benchmark with profiling, and processing the logs and profiling data. For the rest of the workloads, we list the YAML configure files.

UNET3D: 3D Medical Image Segmentation
---------------------------------------
* Reference Implementation: https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch
* Framework: PyTorch
* Dataset: .npz format image files containing a single sample.
* Trains over multiple epochs, performs evaluation on a held-out test set periodically.

.. code-block:: yaml

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
        record_length: 147000000
        keep_files: True
    
    data_reader: 
        data_loader: pytorch

    train:
        epochs: 10
        computation_time: 4.59

    evaluation: 
        eval_time: 11.572
        epochs_between_evals: 2

First, we generate the dataset with ```++workload.workflow.generate=False```

.. code-block :: bash
    
    mpirun -np 8 python src/dlio_benchmark.py workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False

Then, we run the appliation with iostat profiling

.. code-block:: bash
    
    python src/dlio_benchmark.py workload=unet3d ++workload.workflow.profiling=iostat

To run in data parallel mode, one can do

.. code-block:: bash

    mpirun -np 8 src/dlio_benchmark.py workload=unet3d ++workload.workflow.profiling=iostat

This will run the benchmark and produce the following logging output: 

.. code-block:: text

    2022-11-09T17:49:50.271593 Running DLIO with 8 processes [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:91]
    2022-11-09T17:49:50.271870 Running DLIO with 8 processes [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:91]
    2022-11-09T17:49:50.271880 Running DLIO with 8 processes [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:91]
    2022-11-09T17:49:50.271942 Running DLIO with 8 processes [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:91]
    2022-11-09T17:49:50.271958 Running DLIO with 8 processes [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:91]
    2022-11-09T17:49:50.272021 Running DLIO with 8 processes [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:91]
    2022-11-09T17:49:50.272173 Running DLIO with 8 processes [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:91]
    2022-11-09T17:49:50.272300 Running DLIO with 8 processes [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:91]
    2022-11-09T17:49:50.275208 Starting data generation [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:152]
    2022-11-09T17:49:50.275226 Starting data generation [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:152]
    2022-11-09T17:49:50.275214 Starting data generation [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:152]
    2022-11-09T17:49:50.275221 Starting data generation [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:152]
    2022-11-09T17:49:50.275209 Starting data generation [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:152]
    2022-11-09T17:49:50.275223 Starting data generation [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:152]
    2022-11-09T17:49:50.275211 Starting data generation [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:152]
    2022-11-09T17:49:50.275220 Starting data generation [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:152]
    Generated file 0/3662 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/data_generator/npz_generator.py:45]
    Generated file 100/3662 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/data_generator/npz_generator.py:45]
    Generated file 200/3662 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/data_generator/npz_generator.py:45]
    Generated file 300/3662 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/data_generator/npz_generator.py:45]
    Generated file 400/3662 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/data_generator/npz_generator.py:45]
    Generated file 500/3662 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/data_generator/npz_generator.py:45]
    Generated file 600/3662 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/data_generator/npz_generator.py:45]
    Generated file 700/3662 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/data_generator/npz_generator.py:45]
    ...
    ...
    2022-11-09T17:49:52.981932 Generation done [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:154]
    2022-11-09T17:49:53.104430 Generation done [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:154]
    2022-11-09T17:49:53.106440 Profiling Started [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:161]
    2022-11-09T17:49:53.108440 Steps per epoch: 114 = 1 * 3620 / 4 / 8 (samples per file * num files / batch size / comm size) [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:252]
    2022-11-09T17:49:53.108605 Steps per eval: 6 = 1 * 42 / 1 / 8 (samples per file * num files / batch size eval / comm size) [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:255]
    2022-11-09T17:49:53.108742 Starting epoch 1 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:36]
    2022-11-09T17:49:53.164784 Starting block 1 [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:79]
    2022-11-09T17:49:58.117517 Rank 2 processed 4 samples in 4.952726602554321 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:49:58.117616 Rank 0 processed 4 samples in 4.952630043029785 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:49:58.117621 Rank 3 processed 4 samples in 4.952757358551025 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:49:58.117630 Rank 5 processed 4 samples in 4.952760934829712 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:49:58.117621 Rank 1 processed 4 samples in 4.952746152877808 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:49:58.117610 Rank 6 processed 4 samples in 4.952739953994751 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:49:58.117629 Rank 4 processed 4 samples in 4.95275354385376 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:49:58.117626 Rank 7 processed 4 samples in 4.952752113342285 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:50:02.719512 Rank 1 processed 4 samples in 4.600942134857178 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:50:02.719512 Rank 3 processed 4 samples in 4.600916862487793 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:50:02.719511 Rank 7 processed 4 samples in 4.59944748878479 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:50:02.719584 Rank 5 processed 4 samples in 4.601000070571899 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:50:02.719617 Rank 2 processed 4 samples in 4.601523399353027 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:50:02.719634 Rank 4 processed 4 samples in 4.600922107696533 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:50:02.719631 Rank 0 processed 4 samples in 4.601005554199219 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    2022-11-09T17:50:02.719623 Rank 6 processed 4 samples in 4.600902795791626 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:134]
    ...
    ...
    2022-11-09T18:15:31.385725 Rank 4 processed 1 samples in 11.58487319946289 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:146]
    2022-11-09T18:15:31.385794 Rank 0 processed 1 samples in 11.58493447303772 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:146]
    2022-11-09T18:15:31.385758 Rank 5 processed 1 samples in 11.584854364395142 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:146]
    2022-11-09T18:15:31.385741 Rank 7 processed 1 samples in 11.58483624458313 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:146]
    2022-11-09T18:15:31.386264 Rank 1 processed 1 samples in 11.585409879684448 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:146]
    2022-11-09T18:15:31.386779 Rank 3 processed 1 samples in 11.585919857025146 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:146]
    2022-11-09T18:15:31.403301 Ending eval - 6 steps completed in 69.79 s [/home/huihuo.zheng/mlperf/dlio_benchmark/src/utils/statscounter.py:71]
    2022-11-09T18:15:31.411853 profiling stopped [/gpfs/jlse-fs0/projects/datascience/hzheng/mlperf/dlio_benchmark/./src/dlio_benchmark.py:311]


This will generate the logs and profiling data inside hydra_log/${model}/${data}-${time} folder.

.. code-block:: bash

    $ ls hydra_log/unet3d/2022-11-09-17-55-44/
    0_load_and_proc_times.json  3_load_and_proc_times.json  6_load_and_proc_times.json  iostat.json
    1_load_and_proc_times.json  4_load_and_proc_times.json  7_load_and_proc_times.json  per_epoch_stats.json
    2_load_and_proc_times.json  5_load_and_proc_times.json  dlio.log

One can then post processing the data with dlio_postprocessor.py

.. code-block:: bash 

    python src/dlio_postprocessor.py --output-folder hydra_log/unet3d/2022-11-09-17-55-44/

The output is

.. code-block:: text

    ===============Processing DLIO output================
    Job configuration
    output_folder: hydra_log/unet3d/2022-11-09-17-55-44/
    num_proc: 8
    epochs: 2
    batch_size: 4
    do_eval: True
    batch_size_eval: 1
    do_checkpoint: False
    debug: False
    name: unet3d
    WARNING: missing necessary file: hydra_log/unet3d/2022-11-09-17-55-44/iostat.json
    2022-11-09 20:45:29 Generating Report
    2022-11-09 20:45:29 Calculating Loading and Processing Times
    2022-11-09 20:45:29 Reading from hydra_log/unet3d/2022-11-09-17-55-44/0_load_and_proc_times.json
    2022-11-09 20:45:29 Processing loading and processing times for epoch 1
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    2022-11-09 20:45:29 Processing processing times for phase eval
    2022-11-09 20:45:29 Processing loading and processing times for epoch 2
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    2022-11-09 20:45:29 Processing processing times for phase eval
    2022-11-09 20:45:29 Reading from hydra_log/unet3d/2022-11-09-17-55-44/1_load_and_proc_times.json
    2022-11-09 20:45:29 Processing loading and processing times for epoch 1
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    2022-11-09 20:45:29 Processing processing times for phase eval
    2022-11-09 20:45:29 Processing loading and processing times for epoch 2
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    2022-11-09 20:45:29 Processing processing times for phase eval
    2022-11-09 20:45:29 Reading from hydra_log/unet3d/2022-11-09-17-55-44/2_load_and_proc_times.json
    2022-11-09 20:45:29 Processing loading and processing times for epoch 1
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    ...
    ....
    2022-11-09 20:45:29 Reading from hydra_log/unet3d/2022-11-09-17-55-44/3_load_and_proc_times.json
    2022-11-09 20:45:29 Processing loading and processing times for epoch 1
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    2022-11-09 20:45:29 Processing processing times for phase eval
    2022-11-09 20:45:29 Processing loading and processing times for epoch 2
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    2022-11-09 20:45:29 Processing processing times for phase eval
    2022-11-09 20:45:29 Reading from hydra_log/unet3d/2022-11-09-17-55-44/4_load_and_proc_times.json
    2022-11-09 20:45:29 Processing loading and processing times for epoch 1
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    2022-11-09 20:45:29 Processing processing times for phase eval
    2022-11-09 20:45:29 Processing loading and processing times for epoch 2
    2022-11-09 20:45:29 Processing loading times for phase block1
    2022-11-09 20:45:29 Processing loading times for phase eval
    2022-11-09 20:45:29 Processing processing times for phase block1
    ...
    ...
    2022-11-09 20:45:29 Computing overall stats
    2022-11-09 20:45:29 Computing per epoch stats
    2022-11-09 20:45:29 Computing stats for epoch 1 block1
    2022-11-09 20:45:29 Computing stats for epoch 1 eval
    2022-11-09 20:45:29 Computing stats for epoch 2 block1
    2022-11-09 20:45:29 Computing stats for epoch 2 eval
    2022-11-09 20:45:29 Writing report
    2022-11-09 20:45:29 Successfully wrote hydra_log/unet3d/2022-11-09-17-55-44/DLIO_unet3d_report.txt


.. code-block:: yaml

    #contents of DLIO_unet3d_report.txt

    Overall

        Run name:                     unet3d
        Started:                      2022-11-09 17:55:51.466064
        Ended:                        2022-11-09 18:14:21.616347
        Duration (s):                 1110.15
        Num Ranks:                    8
        Batch size (per rank):        4
        Eval batch size:              1


    Detailed Report

    Epoch 1
        Started:             2022-11-09 17:55:51.466064
        Ended:               2022-11-09 18:04:31.698909
        Duration (s):        520.23

        Block 1
            Started:                               2022-11-09 17:55:51.483460
            Ended:                                 2022-11-09 18:04:31.620000
            Duration (s):                          520.14
            Avg loading time / rank (s):           0.55
            Avg processing time / rank (s):        520.09

        Eval 1
            Started:                               2022-11-09 18:04:31.700277
            Ended:                                 2022-11-09 18:05:41.465925
            Duration (s):                          69.77
            Avg loading time / rank (s):           0.21
            Avg processing time / rank (s):        69.72

    ...

BERT: Natural Language Processing Model
---------------------------------------

* Reference Implementation: https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert
* Framework: Tensorflow
* Dataset: Multiple tfrecord files containing many samples each.
* Trains in a single epoch, performs periodic checkpointing of its parameters.

.. code-block:: yaml

    model: bert

    framework: tensorflow

    workflow:
        generate_data: False
        train: True
        debug: False
        checkpoint: True
    
    dataset: 
        data_folder: ./data/bert/
        format: tfrecord
        num_files_train: 500
        num_samples_per_file: 313532
        record-length: 2500
        batch_size: 48

    train:
        computation_time: 0.968
        total_training_steps: 5000
    
    data_reader:
        data_loader: tensorflow
        read_threads: 1
        computation_threads: 8
        transfer_size: 262144

    checkpoint:
        steps_between_checkpoints: 1250
        model_size: 4034713312

CosmoFlow: 3D CNN to Learn the Universe at Scale
----------------------------------------------------
* Reference Implementation: https://github.com/mlcommons/hpc/tree/main/cosmoflow
* Framework: Tensorflow Keras
* Dataset: Multiple tfrecord files containing many samples each.
* Trains in multiple epochs

.. code-block:: yaml

    # contents of cosmoflow.yaml
    model: cosmoflow

    framework: tensorflow

    workflow:
        generate_data: False
        train: True

    dataset:
        data_folder: ./data/cosmoflow
        num_files_train: 1024
        num_samples_per_file: 512
        record_length: 131072
        batch_size: 1

    data_reader:
        data_loader: tensorflow
        computation_threads: 8
        read_threads: 8

    train: 
        epochs: 4

ResNet50: 3D Image classification
-------------------------------------
* Reference Implementation: https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks
* Framework: Tensorflow
* Dataset: ImageNet datasets saved in tfrecords files
* Trains in multiple epochs. 

.. code-block:: yaml

    # contents of resnet50.yaml
    model: resnet50

    framework: tensorflow

    workflow:
        generate_data: False
        train: True

    dataset:
        num_files_train: 1024
        num_samples_per_file: 1024
        record_length: 150528
        data_folder: data/resnet50
        format: tfrecord
    
    data_loader:
        data_loader: tensorflow
        read_threads: 8
        computation_threads: 8