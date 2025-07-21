"""
Copyright (c) 2022, UChicago Argonne, LLC
All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python
import uuid
import shutil
import pytest
import logging
import os
import glob

from mpi4py import MPI

from hydra import initialize_config_dir, compose

from dlio_benchmark.main import DLIOBenchmark
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
import dlio_benchmark

comm = MPI.COMM_WORLD

config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            "dlio_dataset_dimension_test.log", mode="a", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
    format="[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]",
    # logging's max timestamp resolution is msecs, we will pass in usecs in the message
)



def run_benchmark(cfg, storage_root="./"):
    comm.Barrier()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg["workload"])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    return benchmark


@pytest.fixture
def setup_test_env():
    DLIOMPI.get_instance().initialize()
    if comm.rank == 0:
        storage_root = os.path.join("outputs", str(uuid.uuid4()))
    else:
        storage_root = None
    storage_root = comm.bcast(storage_root, root=0)

    if comm.rank == 0:
        if os.path.exists(storage_root):
            shutil.rmtree(storage_root)
        os.makedirs(storage_root, exist_ok=True)

    comm.Barrier()
    yield storage_root
    comm.Barrier()
    if comm.rank == 0:
        shutil.rmtree(storage_root, ignore_errors=True)
    comm.Barrier()
    DLIOMPI.get_instance().finalize()


def check_h5(path):
    import h5py

    f = h5py.File(path, "r")
    keys = list(f.keys())
    keys.remove("labels")
    variable = keys[-1]
    return f[variable].shape, f[variable].dtype, len(keys)


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_dim_based_hdf5_gen_data(setup_test_env, dtype) -> None:
    fmt = "hdf5"
    framework = "pytorch"
    num_dataset_per_record = 3
    shape_per_dataset = (1, 32, 64)
    shape = (num_dataset_per_record * shape_per_dataset[0], *shape_per_dataset[1:])
    storage_root = setup_test_env
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=False",
                "++workload.workflow.generate_data=True",
                # f"++workload.storage.storage_root={storage_root}",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                "++workload.dataset.num_files_train=16",
                "++workload.dataset.num_files_val=0",
                "++workload.dataset.num_subfolders_train=1",
                f"++workload.dataset.format={fmt}",
                f"++workload.dataset.record_dims={list(shape)}",
                f"++workload.dataset.record_element_type={dtype}",
                f"++workload.dataset.hdf5.num_dataset_per_record={num_dataset_per_record}",
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "data", "train", "*.hdf5"))
    if len(paths) == 0:
        pytest.fail("No HDF5 files found")

    chosen_path = paths[0]
    gen_shape, gen_dtype, gen_num_ds = check_h5(chosen_path)

    if comm.rank == 0:
        logging.info("Generated shape: %s", gen_shape)
        logging.info("Generated dtype: %s", gen_dtype)
        logging.info("Number of datasets: %s", gen_num_ds)

    assert shape_per_dataset == gen_shape
    assert dtype == gen_dtype
    assert num_dataset_per_record == gen_num_ds

def check_image(path):
    from PIL import Image

    img = Image.open(path)
    return img.size, img.format


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("fmt", ["png", "jpeg"])
def test_dim_based_image_gen_data(setup_test_env, fmt) -> None:
    framework = "pytorch"
    height = 64
    width = 128
    storage_root = setup_test_env
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=False",
                "++workload.workflow.generate_data=True",
                # f"++workload.storage.storage_root={storage_root}",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                "++workload.dataset.num_files_train=16",
                "++workload.dataset.num_files_val=0",
                "++workload.dataset.num_subfolders_train=1",
                f"++workload.dataset.format={fmt}",
                f"++workload.dataset.record_dims={[height, width]}",
            ],
        )
        print(cfg)
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "data", "train", f"*.{fmt}"))
    if len(paths) == 0:
        pytest.fail(f"No {fmt} files found")

    chosen_path = paths[0]
    gen_shape, gen_format = check_image(chosen_path)

    if comm.rank == 0:
        logging.info("Generated width: %s", gen_shape[0])
        logging.info("Generated height: %s", gen_shape[1])
        logging.info("Generated format: %s", gen_format)

    assert (width, height) == gen_shape
    assert fmt == gen_format.lower()