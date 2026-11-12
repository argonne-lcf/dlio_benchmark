# Running DLIO benchmark with DAOS PyTorch DataLoader

## Prerequisites

 - Python 3.10 or higher
 - openmpi and openmpi-devel packages
 - DAOS client libraries built and installed from [master branch](https://github.com/daos-stack/daos/tree/master/src/client/pydaos/torch)
 - configured and working DAOS agent on the compute nodes


## Getting started


Since `DAOS PyTorch` client was not released outside the `master` branch you'd need to build and install the `pydaos3-package` on the compute node manually (`torch` integration comes with `pydaos` package):


```bash

$: pip install $(DAOS_BUILD_OUTPUT)/install/lib/daos/python

```



## Example of running benchmark with `DAOS` PyTorch client


```bash
# LD_LIBRARY_PATH is needed to load DAOS libraries from build directory
export LD_LIBRARY_PATH=/lus/flare/projects/DAOS_Testing/daos/install/lib64/:$LD_LIBRARY_PATH

mpiexec --np ${NTOTRANKS} -ppn ${NRANKS} --cpu-bind depth -d ${NDEPTH} --no-vni \
	dlio_benchmark workload=daos_pytorch \
	++workload.workflow.generate_data=True  \
	++workload.dataset.daos_pool=DAOS_Testing \
	++workload.dataset.daos_cont=defaults \
	++workload.workflow.checkpoint=True \
	++workload.checkpoint.checkpoint_daos_pool=DAOS_Testing \
	++workload.checkpoint.checkpoint_daos_cont=defaults \
	++workload.checkpoint.checkpoint_folder=/checkpoints \
	++workload.dataset.data_folder=/datasets/small-08 \
	++workload.dataset.num_files_train=80000 \
	++workload.dataset.num_files_eval=10000 \
	++workload.reader.batch_size=32 \
	++workload.reader.read_threads=4 \
	++workload.dataset.record_length_bytes=1048576 \
	++workload.train.epochs=5
```
