# Running DLIO benchmark with DAOS PyTorch DataLoader

## Prerequisites

 - Python 3.10 or higher
 - openmpi and openmpi-devel packages
 - DAOS client libraries built and installed from [master branch](https://github.com/daos-stack/daos/tree/master/src/client/pydaos/torch)
 - configured and working DAOS agent on the compute nodes


## Getting started


```bash
$: pip install mpi4py
$: python3 -m venv ~/venv
$: source ~/venv/bin/activate
$: git clone https://github.com/enaktalabs/dlio_benchmark.git -b daos-pytorch
$: cd dlio_benchmark
$: pip install .
```

Since `DAOS PyTorch` client was not released outside the `master` branch you'd need to build and install the `pydaos3-package` on the compute node manually (`torch` integration comes with `pydaos` package):


```bash

$: pip install $(DAOS_BUILD_OUTPUT)/install/lib/daos/python

````

## Current limitations

`torch.utils.data.Dataset` provides only methods for read-only access to underlying data, which means that it cannot generate benchmark samples, as would be expected for general purpose `FileStorage` implementations.
On the same note, `dlio_benchmark` expects `FileStorage` implementations to walk the directories with samples to build the list of training and validation datasets.
`DaosDfsStorage` implementation provides an ad-hoc solution by building the directory map from the samples filename. Obviously, it's not the fastest nor the most efficient way, but it serves the purpose for the moment.


## Generating samples

By using `dfuse` and local mount:

```bash

$: dfuse /mnt/dfuse test-pool test-container

$: mpirun -np 4 dlio_benchmark workload=dfuse_pytorch ++workload.workflow.generate_data=True ++workload.workflow.train=False ++workload.dataset.data_folder=/mnt/dfuse/dataset ++workload.dataset.num_files_train=5000 ++workload.dataset.record_length=15000 ++workload.dataset.record_length_stdev=0 ++workload.dataset.record_length_resize=0

```


## Running benchmark with `DAOS` PyTorch client


```bash

$: mpirun -np 4 dlio_benchmark workload=daos_pytorch ++workload.workflow.generate_data=False ++workload.workflow.checkpoint=False ++workload.dataset.num_files_train=5000 ++workload.dataset.record_length=15000 ++workload.train.epochs=3  ++workload.dataset.daos_pool=test-pool ++workload.dataset.daos_cont=test-container ++workload.reader.batch_size=32 ++workload.reader.read_threads=4 ++workload.dataset.data_folder=/dataset

```

Note that, while generating samples with dfuse `workload.dataset.data_folder=/mnt/dfuse/dataset` was pointing to `local` path whereas during training `workload.dataset.data_folder=/dataset` specifies the path inside POSIX container.
