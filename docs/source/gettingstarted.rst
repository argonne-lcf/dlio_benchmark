In this section, we present in details how to run DLIO

Installation
============
DLIO itself should run directly without installation. However it depends on other python packages which are listed in requirements.txt in the github repo. 

.. code-block::

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install -r requirements.txt 
    python ./src/dlio_benchmark.py --help

Running the benchmark
=====================

A DLIO run is split in 3 phases:

* Generate synthetic data DLIO will use
* Run the benchmark using the previously generated data
* Post-process the results to generate a report

The configurations of a workload can be specified through a yaml file. Examples of yaml files can be find in ./configs/workload/.

One can specify specify workload through "workload=" option in the command line. The configuration can be overridden through commandline in the hyra framework (e.g.++workload.framework=tensorflow).

For example, to run the unet3d benchark, one can do

.. code-block::

    mpirun -np 8 python src/dlio_benchmark.py workload=unet3d

This will both generate the dataset and perform benchmark.

One can separate the data generation part and training part as

**Generate data**

.. code-block::

    mpirun -np 8 python src/dlio_benchmark.py workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False ++workload.workflow.evaluation=False

**Running benchmark**

.. code-block::

    mpirun -np 8 python src/dlio_benchmark.py workload=unet3d ++workload.workflow.generate_data=False ++workload.workflow.train=True ++workload.workflow.evaluation=True

All the outputs will be stored in hydra_log/unet3d/$DATE-$TIME folder. To post process the data, one can do

.. code-block::

    python3 src/dlio_postprocessor.py --output_folder=hydra_log/unet3d/$DATE-$TIME

This will generate DLIO_$model_report.txt inside the output folder.

Examples
=============
