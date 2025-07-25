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
import pytest
import logging
import os
import glob
from datetime import datetime

from mpi4py import MPI

import numpy as np

from hydra import initialize_config_dir, compose

from dlio_benchmark.main import DLIOBenchmark
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
import dlio_benchmark

comm = MPI.COMM_WORLD

config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

logging.basicConfig(
    level=print,
    handlers=[
        logging.FileHandler(
            "dlio_dataset_dimension_test.log", mode="a", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
    format="[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]",
    # logging's max timestamp resolution is msecs, we will pass in usecs in the message
)

def delete_folder(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)

def run_benchmark(cfg):
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
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        storage_root = os.path.join("outputs", f"{now}-{str(uuid.uuid4())}")
    else:
        storage_root = None
    storage_root = comm.bcast(storage_root, root=0)

    if comm.rank == 0:
        if os.path.exists(storage_root):
            delete_folder(storage_root)
        os.makedirs(storage_root, exist_ok=True)

    comm.Barrier()
    yield storage_root
    comm.Barrier()
    if comm.rank == 0:
        delete_folder(storage_root)
    comm.Barrier()


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
    num_data_pp = 2
    num_data = num_data_pp * comm.size
    storage_root = setup_test_env
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=False",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={num_data}",
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

    chosen_path = paths[comm.rank % len(paths)]
    gen_shape, gen_dtype, gen_num_ds = check_h5(chosen_path)

    if comm.rank == 0:
        print(f"Generated shape: {gen_shape}")
        print(f"Generated dtype: {gen_dtype}")
        print(f"Number of datasets: {gen_num_ds}")

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
    num_data_pp = 2
    num_data = num_data_pp * comm.size
    storage_root = setup_test_env
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=False",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={num_data}",
                "++workload.dataset.num_files_val=0",
                "++workload.dataset.num_subfolders_train=1",
                f"++workload.dataset.format={fmt}",
                f"++workload.dataset.record_dims={[height, width]}",
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "data", "train", f"*.{fmt}"))
    if len(paths) == 0:
        pytest.fail(f"No {fmt} files found")

    chosen_path = paths[comm.rank % len(paths)]
    gen_shape, gen_format = check_image(chosen_path)

    if comm.rank == 0:
        print(f"Generated width: {gen_shape[0]}")
        print(f"Generated height: {gen_shape[1]}")
        print(f"Generated format: {gen_format}")

    assert (width, height) == gen_shape
    assert fmt == gen_format.lower()

def check_np(path, fmt):
    if fmt == "npy":
        data = np.load(path)
        return data.shape, data.dtype
    elif fmt == "npz":
        data = np.load(path)
        return data["x"].shape, data["x"].dtype
    else:
        raise ValueError(f"Unsupported format: {fmt}")


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("fmt, dtype", [("npz", "int32"), ("npz", "float32"), ("npy", "int32"), ("npy", "float32")])
def test_dim_based_np_gen_data(setup_test_env, fmt, dtype) -> None:
    framework = "pytorch"
    num_samples_per_file = 1
    shape = (64, 128)
    num_data_pp = 2
    num_data = num_data_pp * comm.size
    final_shape = (*shape, num_samples_per_file)
    storage_root = setup_test_env
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=False",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={num_data}",
                "++workload.dataset.num_files_val=0",
                "++workload.dataset.num_subfolders_train=1",
                f"++workload.dataset.num_samples_per_file={num_samples_per_file}",
                f"++workload.dataset.format={fmt}",
                f"++workload.dataset.record_element_type={dtype}",
                f"++workload.dataset.record_dims={list(shape)}",
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "data", "train", f"*.{fmt}"))
    if len(paths) == 0:
        pytest.fail(f"No {fmt} files found")

    chosen_path = paths[comm.rank % len(paths)]
    gen_shape, gen_format = check_np(chosen_path, fmt=fmt)

    if comm.rank == 0:
        print(f"Generated shape: {gen_shape}")
        print(f"Generated format: {gen_format}")

    assert final_shape == gen_shape
    assert np.dtype(dtype).itemsize == gen_format.itemsize

def check_tfrecord(paths):
    import tensorflow as tf
    dataset = tf.data.TFRecordDataset(paths)

    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }

    for data in dataset.take(1):
        parsed = tf.io.parse_example(data, features)
        record_length_bytes = (
            tf.strings.length(parsed["image"], unit="BYTE").numpy().item()
        )        
        return record_length_bytes

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("dtype", ["int16", "float32", "uint8"])
def test_dim_based_tfrecord_gen_data(setup_test_env, dtype) -> None:
    framework = "tensorflow"
    shape = (64, 128)
    storage_root = setup_test_env
    num_data_pp = 2
    num_data = num_data_pp * comm.size
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=False",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={num_data}",
                "++workload.dataset.num_files_val=0",
                "++workload.dataset.num_subfolders_train=1",
                "++workload.dataset.format=tfrecord",
                f"++workload.dataset.record_element_type={dtype}",
                f"++workload.dataset.record_dims={list(shape)}",
            ],
        )
        run_benchmark(cfg)

    train_data_dir = os.path.join(storage_root, "data", "train")
    paths = glob.glob(os.path.join(train_data_dir, "*.tfrecord"))
    if len(paths) == 0:
        pytest.fail("No tfrecord files found")

    gen_bytes = check_tfrecord(paths)

    if comm.rank == 0:
        print(f"Generated bytes: {gen_bytes}")

    assert np.prod(shape) * np.dtype(dtype).itemsize == gen_bytes

# @ray: this code is taken from dlio_benchmark/reader/indexed_binary_reader.py
# if that file is changed this code may need to be updated
def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a

# @ray: this code is taken from dlio_benchmark/reader/indexed_binary_reader.py
# if that file is changed this code may need to be updated
def index_file_path_off(prefix_path):
    return prefix_path + '.off.idx'

# @ray: this code is taken from dlio_benchmark/reader/indexed_binary_reader.py
# if that file is changed this code may need to be updated
def index_file_path_size(prefix_path):
    return prefix_path + '.sz.idx'

# @ray: this code is taken from dlio_benchmark/reader/indexed_binary_reader.py
# if that file is changed this code may need to be updated
def get_indexed_metadata(path, num_samples_per_file):
    offset_file = index_file_path_off(path)
    sz_file = index_file_path_size(path)
    with open(offset_file, 'rb') as f:
        offsets = read_longs(f, num_samples_per_file)
    with open(sz_file, 'rb') as f:
        sizes = read_longs(f, num_samples_per_file)
    return offsets, sizes

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("dtype, num_samples_per_file", [
    (dtype, num_samples_per_file)
    for dtype in ["int16", "float32"]  # data types
    for num_samples_per_file in [1, 2, 3]  # even and odd
])
def test_dim_based_indexed_gen_data(setup_test_env, dtype, num_samples_per_file) -> None:
    framework = "pytorch"
    shape = (64, 128)
    storage_root = setup_test_env
    num_data_pp = 2
    num_data = num_data_pp * comm.size
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=False",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={num_data}",
                "++workload.dataset.num_files_val=0",
                "++workload.dataset.num_subfolders_train=1",
                "++workload.dataset.format=indexed_binary",
                f"++workload.dataset.num_samples_per_file={num_samples_per_file}",
                f"++workload.dataset.record_element_type={dtype}",
                f"++workload.dataset.record_dims={list(shape)}",
            ],
        )
        run_benchmark(cfg)

    train_data_dir = os.path.join(storage_root, "data", "train")
    paths = glob.glob(os.path.join(train_data_dir, "*.indexed_binary"))
    if len(paths) == 0:
        pytest.fail("No indexed_binary files found")

    chosen_path = paths[comm.rank % len(paths)]
    offsets, sizes = get_indexed_metadata(chosen_path, num_samples_per_file)

    assert len(offsets) == num_samples_per_file
    assert len(sizes) == num_samples_per_file

    if comm.rank == 0:
        print(f"Generated offsets: {offsets}")        
        print(f"Generated sizes: {sizes}")

    sample_size = np.prod(shape) * np.dtype(dtype).itemsize
    sample_size = sample_size.item()

    with open(chosen_path, "rb") as f:
        for i in range(len(offsets)):
            f.seek(offsets[i])
            data = f.read(sizes[i])
            assert len(data) == sizes[i]
            if comm.rank == 0:
                print(f"Read data of size {len(data)}")
            assert len(data) == sample_size, f"Sample size mismatch: {len(data)} != {sample_size}"