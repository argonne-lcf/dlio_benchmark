# Deep Learning I/O (DLIO) Benchmark
![test status](https://github.com/russfellows/dlio_benchmark/actions/workflows/fast-ci.yml/badge.svg)
![version](https://img.shields.io/badge/version-3.0.1-blue)
![python](https://img.shields.io/badge/python-3.12-blue)
![tests](https://img.shields.io/badge/tests-112%20passed-brightgreen)

> **This is the [russfellows](https://github.com/russfellows/dlio_benchmark) enhanced fork of the
> [MLCommons DLIO benchmark](https://github.com/mlcommons/DLIO_local_changes), which is itself
> derived from the original [Argonne DLIO benchmark](https://github.com/argonne-lcf/dlio_benchmark).**
>
> This fork is **substantially ahead** of the original Argonne upstream.  It includes all
> MLCommons enhancements plus significant additional work described below.  The architecture,
> performance, and feature set have diverged enough that this should be considered a major new
> release rather than a minor patch on top of the 2021 code base.

## What's New vs Upstream

### Data Generation
- **dgen-py zero-copy backend** — Rust-backed data generator is the new default. Generates random tensor data at >15 GB/s with zero heap allocation overhead, replacing the NumPy-based generator that was ~155× slower per sample. Controlled through `dgen-py` Python bindings.
- **Parallel data generation** (`write_threads`) — files are generated concurrently via `ThreadPoolExecutor`, cutting multi-file datagen wall-clock time proportionally to thread count.
- **Reproducible seeding** — `gen_random_tensor(seed=N)` now correctly seeds the dgen fast path; upstream silently ignored the seed.

### Data Formats
- **Parquet format** (new) — full schema-driven generation and reading. Supports mixed-dtype column schemas (`int32`, `float32`, `float16`, `uint8`, `bool`, etc.) with configurable embedding sizes, LZ4/ZSTD compression, and per-column filtering on reads. Legacy single-column mode is preserved for backward compatibility.

### Storage Backends
- **S3 / S3-compatible object storage** (new) — three client libraries supported: [s3dlio](https://github.com/russfellows/s3dlio) (recommended, Rust-backed, multi-endpoint), [s3torchconnector](https://github.com/awslabs/s3-connector-for-pytorch) (PyTorch only), and the [MinIO Python SDK](https://min.io/docs/minio/linux/developers/python/API.html).
- **Multi-endpoint load balancing** — `S3_ENDPOINT_URIS` distributes datagen write load across multiple S3 servers, one endpoint per MPI rank (round-robin). Eliminates single-node bottlenecks for large-scale data generation.
- **Storage env-var overrides** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_REGION`, `DLIO_BUCKET`, `DLIO_STORAGE_TYPE`, and `DLIO_STORAGE_LIBRARY` are all read automatically; no YAML changes needed for credential injection.
- **Post-generation settle guard** — configurable `post_generation_settle_seconds` for eventual-consistency object stores that need time to propagate written objects before training reads begin.

### Correctness Fixes
- **DLRM OOM fix** — replaced the materialized 168 M-entry index dict with a constant-memory `VirtualIndexMap`, eliminating the SIGKILL that occurred on configurations with large datasets.
- **MPI topology auto-sizing** — `read_threads` now divides by `ranks_per_node()` instead of total `comm_size`, giving correct per-node thread counts on multi-node runs.
- **StatsCounter metrics** — fixed three bugs: negative throughput from a magic `(len - 2)` constant, unguarded division when the metric window is empty, and a dict-stomping bug in `batch_processed` that silently corrupted per-block data.

### Testing
- **112-test suite** — comprehensive `pytest`-based CI covering enumerations, config, all generator formats, parquet reader metadata and caching, StatsCounter metrics accuracy, issue regression guards, MPI smoke tests, and end-to-end smoke tests. The original upstream had no automated test suite.
- **Fast CI script** (`tests/run_fast_ci.sh`) — runs the full suite locally in ~15 seconds; mirrors `.github/workflows/fast-ci.yml` exactly.

### Modernization
- **Python 3.12** only — drops support for EOL Python versions; fully typed and lint-clean.
- **uv** package management — `pyproject.toml`-first with `uv.lock` for reproducible installs.

---

## Overview

DLIO is an I/O benchmark for Deep Learning. DLIO is aimed at emulating the I/O behavior of various deep learning applications. The benchmark is delivered as an executable that can be configured for various I/O patterns. It uses a modular design to incorporate more data loaders, data formats, datasets, and configuration parameters. It emulates modern deep learning applications using Benchmark Runner, Data Generator, Format Handler, and I/O Profiler modules.

DLIO supports multiple storage backends out of the box:
- **Local filesystem** — the default, for NFS, Lustre, GPFS, and local NVMe
- **AWS S3 / S3-compatible object storage** — via [s3dlio](https://github.com/russfellows/s3dlio), [s3torchconnector](https://github.com/awslabs/s3-connector-for-pytorch), or the [MinIO Python SDK](https://min.io/docs/minio/linux/developers/python/API.html)
- **AIStore** — via the native AIStore Python SDK

Object storage backends are configured through the `storage:` block in the workload YAML file (see [Object Storage Configuration](#object-storage-configuration) below).

## Installation and running DLIO
### Bare metal installation 

```bash
git clone https://github.com/russfellows/dlio_benchmark
cd dlio_benchmark/
pip install .
dlio_benchmark ++workload.workflow.generate_data=True
```

### Bare metal installation with AIStore support

```bash
git clone https://github.com/russfellows/dlio_benchmark
cd dlio_benchmark/
pip install .[aistore]
```

### Bare metal installation with S3 / object storage support

For S3-compatible object storage (AWS S3, MinIO, Vast Data, etc.) install one or more of the supported storage libraries alongside DLIO:

```bash
git clone https://github.com/russfellows/dlio_benchmark
cd dlio_benchmark/
pip install .

# Choose one (or more) S3 client libraries:
pip install s3dlio                   # recommended — high-performance Rust-backed S3 client
pip install s3torchconnector         # AWS S3 Connector for PyTorch (PyTorch only)
pip install minio                    # MinIO Python SDK
```

The storage library to use is selected per-workload via `storage.storage_options.storage_library` in the YAML config (see [Object Storage Configuration](#object-storage-configuration)).

### Bare metal installation with profiler

```bash
git clone https://github.com/russfellows/dlio_benchmark
cd dlio_benchmark/
pip install .[pydftracer]
```

## Container
```bash
git clone https://github.com/russfellows/dlio_benchmark
cd dlio_benchmark/
docker build -t dlio .
docker run -t dlio dlio_benchmark ++workload.workflow.generate_data=True
``` 

You can also pull rebuilt container from docker hub (might not reflect the most recent change of the code): 
```bash
docker pull docker.io/zhenghh04/dlio:latest
docker run -t docker.io/zhenghh04/dlio:latest dlio_benchmark ++workload.workflow.generate_data=True
```
If your running on a different architecture, refer to the Dockerfile to build the dlio_benchmark container from scratch.

One can also run interactively inside the container
```bash
docker run -t docker.io/zhenghh04/dlio:latest /bin/bash
root@30358dd47935:/workspace/dlio$ dlio_benchmark ++workload.workflow.generate_data=True
```

## PowerPC
PowerPC requires installation through anaconda.
```bash
# Setup required channels
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

# create and activate environment
conda env create --prefix ./dlio_env_ppc --file environment-ppc.yaml --force
conda activate ./dlio_env_ppc
# install other dependencies
python -m pip install .
```

## Lassen, LLNL
For specific instructions on how to install and run the benchmark on Lassen please refer to: [Install Lassen](https://dlio-benchmark.readthedocs.io/en/latest/instruction_lassen.html)

## Running the benchmark

A DLIO run is split in 3 phases: 
- Generate synthetic data that DLIO will use
- Run the benchmark using the previously generated data
- Post-process the results to generate a report

The configurations of a workload can be specified through a yaml file. Examples of yaml files can be found in [dlio_benchmark/configs/workload/](./dlio_benchmark/configs/workload). 

One can specify the workload through the ```workload=``` option on the command line. Specific configuration fields can then be overridden following the ```hydra``` framework convention (e.g. ```++workload.framework=tensorflow```). 

First, generate the data
  ```bash
  mpirun -np 8 dlio_benchmark workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False
  ```
If possible, one can flush the filesystem caches in order to properly capture device I/O
  ```bash
  sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
  ```
Finally, run the benchmark
  ```bash
  mpirun -np 8 dlio_benchmark workload=unet3d
  ```
Finally, run the benchmark with Tracer
  ```bash
  export DFTRACER_ENABLE=1
  export DFTRACER_INC_METADATA=1
  mpirun -np 8 dlio_benchmark workload=unet3d
  ```

All the outputs will be stored in ```hydra_log/unet3d/$DATE-$TIME``` folder. To post process the data, one can do
```bash 
dlio_postprocessor --output-folder hydra_log/unet3d/$DATE-$TIME
```
This will generate ```DLIO_$model_report.txt``` in the output folder. 

## Workload YAML configuration file 
Workload characteristics are specified by a YAML configuration file. Below is an example of a YAML file for the UNet3D workload which is used for 3D image segmentation. 

```
# contents of unet3d.yaml
model: 
  name: unet3d
  model_size: 499153191

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
  record_length_bytes: 146600628
  record_length_bytes_stdev: 68341808
  record_length_bytes_resize: 2097152
  
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
```

The full list of configurations can be found in: https://argonne-lcf.github.io/dlio_benchmark/config.html

---

## Object Storage Configuration

Object storage is enabled by adding a `storage:` block to the workload YAML.  The `storage_type: s3` value activates the S3 backend; a `storage_library` field selects the underlying client library.

### Supported storage libraries

| `storage_library` | Description | Framework support |
|---|---|---|
| `s3dlio` | High-performance Rust-backed client via [s3dlio](https://github.com/russfellows/s3dlio). Parallel GET, range optimization, multi-endpoint load balancing. | PyTorch + TensorFlow |
| `s3torchconnector` | AWS S3 Connector for PyTorch — streaming single-file GET. | PyTorch only |
| `minio` | MinIO Python SDK via `ThreadPoolExecutor`. | PyTorch + TensorFlow |

### Example: UNet3D with S3 object storage

```yaml
# contents of unet3d_s3.yaml
model:
  name: unet3d
  model_size: 499153191

framework: pytorch

workflow:
  generate_data: False
  train: True
  checkpoint: False

dataset:
  data_folder: my-bucket/unet3d   # path within the bucket
  format: npz
  num_files_train: 168
  num_samples_per_file: 1
  record_length_bytes: 146600628
  record_length_bytes_stdev: 68341808
  record_length_bytes_resize: 2097152

storage:
  storage_type: s3
  storage_root: my-bucket         # S3 bucket name
  storage_library: s3dlio         # client library (s3dlio | s3torchconnector | minio)
  storage_options:
    endpoint_url: http://your-s3-host:9000   # omit for AWS; required for MinIO etc.
    region: us-east-1
    # Credentials come from environment variables:
    #   export AWS_ACCESS_KEY_ID=...
    #   export AWS_SECRET_ACCESS_KEY=...

reader:
  data_loader: pytorch
  batch_size: 7
  read_threads: 4
  file_shuffle: seed
  sample_shuffle: seed
  # Required when using s3dlio with PyTorch multiprocessing:
  multiprocessing_context: spawn

train:
  epochs: 5
  computation_time: 0.323
```

### Running with object storage

Set credentials via environment variables before running:

```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_ENDPOINT_URL=http://your-s3-host:9000   # for non-AWS endpoints

# Generate data into S3
mpirun -np 8 dlio_benchmark workload=unet3d_s3 ++workload.workflow.generate_data=True ++workload.workflow.train=False

# Run benchmark from S3
mpirun -np 8 dlio_benchmark workload=unet3d_s3
```

### Multi-Endpoint Distribution (datagen)

All three storage libraries support distributing datagen write load across
multiple S3-compatible servers.  Set `S3_ENDPOINT_URIS` to a comma-separated
list of endpoints; each MPI rank is automatically assigned one endpoint using
round-robin (`endpoint[rank % num_endpoints]`).

```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export S3_ENDPOINT_URIS='http://storage1:9000,http://storage2:9000,http://storage3:9000,http://storage4:9000'

# Generate data — rank 0 → storage1, rank 1 → storage2, rank 2 → storage3, rank 3 → storage4
mpirun -np 16 dlio_benchmark workload=retinanet_datagen \
  ++workload.storage.storage_options.storage_library=s3dlio
```

**How it works** (`dlio_benchmark/storage/obj_store_lib.py`):

1. At init, the storage backend reads `S3_ENDPOINT_URIS` (comma-separated list).
2. If running under MPI it reads the process rank from `OMPI_COMM_WORLD_RANK`
   (Open MPI), `PMI_RANK` (MPICH / Intel MPI), `MV2_COMM_WORLD_RANK` (MVAPICH2),
   or `SLURM_PROCID` (Slurm `srun`).
3. The selected endpoint is `endpoints[rank % len(endpoints)]`, so any number of
   ranks distributes evenly regardless of whether ranks divide cleanly into
   endpoint count.
4. If no rank variable is found (single-process run), the first endpoint is used
   with a warning.

**Measured distribution** (April 2026, 2 × s3-ultra local servers, NP=2, 2000 files):

| Library | EP1 objects | EP2 objects | Balance |
|---|---|---|---|
| s3dlio | 1000 | 1000 | 100% |
| minio | 1000 | 1000 | 100% |
| s3torchconnector | 1000 | 1000 | 100% |

**s3dlio additionally** supports a native `MultiEndpointStore` (per-request
round-robin within a single process), usable directly via the Python API:

```python
import s3dlio, os

os.environ["AWS_ACCESS_KEY_ID"] = "key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "secret"

store = s3dlio.create_multi_endpoint_store(
    uris=["s3://storage1:9000/bucket/", "s3://storage2:9000/bucket/"],
    strategy="round_robin",   # or "least_connections"
)

# PUT distributes across endpoints per-request — no MPI needed
import asyncio
asyncio.run(store.put("data/file.bin", data_bytes))

# Per-endpoint stats
for ep in store.get_endpoint_stats():
    print(ep["uri"], ep["total_requests"], ep["bytes_written"])
```

**Important limitation**: Objects written to one server are not replicated to
other servers (no shared backend). A dataset split across endpoints must be
read back with the same endpoint-per-rank assignment used during datagen.
`S3_ENDPOINT_URIS` must be set identically for both datagen and training runs.

Pre-built S3 workload configs matching MLPerf Storage GPU profiles are available in [dlio_benchmark/configs/workload/](./dlio_benchmark/configs/workload/) (e.g. `unet3d_h100_s3.yaml`, `unet3d_a100_s3.yaml`, `unet3d_v100_s3.yaml`).

### Timing correctness with object storage

The training loop timing is **not affected** by switching to object storage. The measurement sequence (`start_loading` → batch delivery → `batch_loaded` → GPU sleep → `batch_processed`) is identical to local filesystem runs. Object storage I/O happens inside PyTorch DataLoader worker processes during the GPU computation sleep, exactly as local file reads do. See [docs/DLIO-Object-Storage_Analysis.md](./docs/DLIO-Object-Storage_Analysis.md) for a detailed analysis.

---

The YAML file is loaded through hydra (https://hydra.cc/). The default setting are overridden by the configurations loaded from the YAML file. One can override the configuration through command line (https://hydra.cc/docs/advanced/override_grammar/basic/). 

## Current Limitations and Future Work

* DLIO currently assumes the samples to always be 2D images, even though one can set the size of each sample through ```--record_length```. We expect the shape of the sample to have minimal impact to the I/O itself. This yet to be validated for case by case perspective. We plan to add option to allow specifying the shape of the sample. 

* We assume the data/label pairs are stored in the same file. Storing data and labels in separate files will be supported in future.

* File format support: we only support tfrecord, hdf5, npz, csv, jpg, jpeg formats. Other data formats can be extended.

* Storage backend support: we support local filesystem (`local_fs`), AWS S3 and S3-compatible object stores (`s3`), and AIStore (`aistore`). For S3 storage, three client libraries are available: [s3dlio](https://github.com/russfellows/s3dlio) (recommended), [s3torchconnector](https://github.com/awslabs/s3-connector-for-pytorch) (PyTorch only), and the [MinIO SDK](https://min.io/docs/minio/linux/developers/python/API.html). Other storage backends can be extended.

* Data Loader support: we support reading datasets using TensorFlow tf.data data loader, PyTorch DataLoader, and a set of custom data readers implemented in ./reader. For TensorFlow tf.data data loader, PyTorch DataLoader  
  - We have complete support for tfrecord format in TensorFlow data loader. 
  - For npz, jpg, jpeg, hdf5, we currently only support one sample per file case. In other words, each sample is stored in an independent file. Multiple samples per file case will be supported in future. 

## How to contribute

This repository is the [russfellows](https://github.com/russfellows/dlio_benchmark) enhanced fork.
Contributions and issues should be filed against this fork rather than the original Argonne upstream.

We welcome contributions in the following areas:

* Support for new workloads — submit a YAML config and open a PR.
* New data loaders (DALI, MxNet, etc.)
* New frameworks (MxNet, JAX, etc.)
* Additional storage backends (GCS, Azure Blob, etc.)
* New data formats
* Bug reports and correctness fixes

For issues related to the MLCommons Storage benchmark specification, also see the
[MLCommons DLIO fork](https://github.com/mlcommons/DLIO_local_changes).

For the original Argonne LCF benchmark, see
[argonne-lcf/dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark)
or contact the ALCF DLIO team (Huihuo Zheng, huihuo.zheng@anl.gov).

## Citation and Reference
The original CCGrid'21 paper describes the design and implementation of DLIO code. Please cite this paper if you use DLIO for your research. 

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
Copyright (c) 2025, UChicago Argonne, LLC
All Rights Reserved

If you have questions about your rights to use or distribute this software, please contact Argonne Intellectual Property Office at partners@anl.gov

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
