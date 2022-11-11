Running DLIO
======================
A DLIO run is split in 3 phases:

1. Generate synthetic data DLIO will use
2. Run the benchmark using the previously generated data
3. Post-process the results to generate a report

One can specify the workload through ```workload=WORKLOAD``` option in the command line. This will read in corresponding configuration files in the `workload`_ folder.  The configuration can be overridden through command line following the hyra syntax (e.g.++workload.framework=tensorflow).

1 and 2 can be done either together or in separate. This is controlled by ```workflow.generate_data``` and ```workload.train``` in the configure file. If ```workflow.generate_data```, ```workflow.train```are all set to be ``True``, it will generate the data and then run the benchark. However, we always suggest to run it seperately, to avoid caching effect, and to avoid I/O profiling in the data generation part. 

'''''''''''''''''''''''
Generate data
'''''''''''''''''''''''

.. code-block:: bash

    mpirun -np 8 python src/dlio_benchmark.py workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False 

In this case, we override ```workflow.generate_data``` and ```workflow.train``` in the configuration to perform the data generation.  

''''''''''''''''''''''
Running benchmark
''''''''''''''''''''''

.. code-block:: bash 

    mpirun -np 8 python src/dlio_benchmark.py workload=unet3d ++workload.workflow.generate_data=False ++workload.workflow.train=True ++workload.workflow.evaluation=True

In this case, we set ```workflow.generate_data=False```, so it will perform training and evaluation with the data generated previously. 

'''''''''''''''''
Post processing
'''''''''''''''''
After running the benchmark, the outputs will be stored in the ```hydra_log/unet3d/$DATE-$TIME``` folder created by hydra. The folder will contains: (1) logging output from the run; (2) profiling outputs; (3) YAML config files: `config.yaml`, `overrides.yaml`, and `hydra.yaml`. The workload configuration file is included in `config.yaml`. Any overrides in the command line are included in `overrides.yaml`. 

To post process the data, one only need to specify the output folder. All the other setups will be automatically read from `config.yaml` inside the folder. 

.. code-block:: bash 

    python3 src/dlio_postprocessor.py --output_folder=hydra_log/unet3d/$DATE-$TIME

This will generate DLIO_$model_report.txt inside the output folder.

.. _workload: https://github.com/argonne-lcf/dlio_benchmark/blob/main/configs/workload
.. _unet3d.yaml: https://github.com/argonne-lcf/dlio_benchmark/blob/main/configs/workload/unet3d.yaml