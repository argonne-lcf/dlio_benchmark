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
from hydra import initialize, initialize_config_dir, compose
from omegaconf import OmegaConf
import unittest
from datetime import datetime
import uuid
from io import BytesIO
import glob
from mpi4py import MPI
import pathlib

comm = MPI.COMM_WORLD

import pytest
import time
import subprocess
import logging
import os
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
import dlio_benchmark

from unittest.mock import patch
try:
    from s3torchconnector._s3client import MockS3Client
except ImportError as e:
    MockS3Client = None
from urllib.parse import urlparse

config_dir=os.path.dirname(dlio_benchmark.__file__)+"/configs/"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_benchmark_test.log", mode="a", encoding='utf-8'),
        logging.StreamHandler()
    ], format='[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
    # logging's max timestamp resolution is msecs, we will pass in usecs in the message
)

from dlio_benchmark.main import DLIOBenchmark, set_dftracer_initialize, set_dftracer_finalize

def finalize():
    # DLIOMPI.get_instance().finalize()
    pass


def clean_s3(mock_client, bucket: str, prefixes: list[str]) -> None:
    comm.Barrier()
    if comm.rank == 0:
        for prefix in prefixes:
            keys = mock_client.list_objects(bucket, prefix)
            for key in keys:
                mock_client.remove_object(key)
    comm.Barrier()

def get_s3_prefixes_from_uri(uri: str, subdirs=("train", "valid")):
    parsed = urlparse(uri)
    base_prefix = parsed.path.lstrip("/")
    return [f"{base_prefix}/{subdir}" for subdir in subdirs]

def run_benchmark(cfg, verify=True):
    comm.Barrier()
    t0 = time.time()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg["workload"])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    t1 = time.time()
    if (comm.rank==0):
        logging.info("Time for the benchmark: %.10f" %(t1-t0))
        if (verify):
            assert(len(glob.glob(benchmark.output_folder+"./*_output.json"))==benchmark.comm_size)    
    return benchmark

class SafeMockS3Client:
    def __init__(self, storage):
        self.storage = storage

    def get_object(self, bucket, key, start=None, end=None):
        if key.startswith("s3://"):
            key = key[len("s3://"):]
            key = key.split("/", 1)[1]
        elif key.startswith(bucket + "/"):
            key = key[len(bucket) + 1:]
        data = self.storage.get(key, b"")
        if start is not None and end is not None:
            return BytesIO(data[start:end+1])
        return BytesIO(data)

    def put_object(self, bucket, key, storage_class=None):
        if key.startswith("s3://"):
            key = key[len("s3://"):]
            key = key.split("/", 1)[1]
        return MockS3Writer(key, self.storage)

    def list_objects(self, bucket, prefix="", delimiter=None, max_keys=None):
        parsed = urlparse(prefix)
        if parsed.scheme == 's3':
            prefix = parsed.path.lstrip('/')
        keys = [k for k in self.storage.keys() if k.startswith(prefix)]
        if max_keys is not None:
            keys = keys[:max_keys]
        stripped_keys = [k[len(prefix):].lstrip("/") if k.startswith(prefix) else k for k in keys]
        return [MockListObjectsResult([MockObjectInfo(k) for k in stripped_keys])]

class MockS3Writer:
    def __init__(self, key, storage):
        self.key = key
        self.storage = storage
        self.buffer = bytearray()

    def write(self, data):
        self.buffer.extend(data)

    def close(self):
        self.storage[self.key] = bytes(self.buffer)

class MockObjectInfo:
    def __init__(self, key):
        self.key = key

class MockListObjectsResult:
    def __init__(self, object_info_list):
        self.object_info = object_info_list

@pytest.fixture
def setup_test_env():
    DLIOMPI.get_instance().initialize()
    if comm.rank == 0:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        storage_root = f"s3-test-bucket-{now}-{str(uuid.uuid4())}"
        storage_type = "s3"
    else:
        storage_root = None
        storage_type = None
        mock_client = None

    storage_root = comm.bcast(storage_root, root=0)
    storage_type = comm.bcast(storage_type, root=0)

    # Only rank 0 initializes the mock storage
    if comm.rank == 0:
        # Shared in-memory mock storage
        mock_storage = {}

        # Create mock client
        mock_client = MockS3Client(region="us-east-1", bucket=storage_root)
        mock_client.storage = mock_storage

        # Simulate bucket existence
        mock_client.add_object("init.txt", b"bucket initialized")
        mock_storage = mock_client.storage
    else:
        mock_storage = None
        mock_client = MockS3Client(region="us-east-1", bucket=storage_root)

    # Broadcast the mock_storage dictionary to all ranks
    mock_storage = comm.bcast(mock_storage, root=0)
    mock_client.storage = mock_storage

    # Patch internal client builder to return the same mock
    mock_client._client_builder = lambda: mock_client._mock_client

    # Patch put_object and get_object to simulate S3 behavior
    def mock_put_object(bucket, key, storage_class=None):
        if key.startswith("s3://"):
            key = key[len("s3://"):]
            key = key.split("/", 1)[1]
        return MockS3Writer(key, mock_storage)

    def mock_get_object(bucket, key, start=None, end=None):
        if key.startswith("s3://"):
            key = key[len("s3://"):]
            key = key.split("/", 1)[1]
        elif key.startswith(bucket + "/"):
            key = key[len(bucket) + 1:]  # removes bucket name if it's prepended manually

        data = mock_storage.get(key, b"")
        if start is not None and end is not None:
            return BytesIO(data[start:end+1])
        return BytesIO(data)

    def mock_list_objects(bucket, prefix="", delimiter=None, max_keys=None):
        # Just use prefix directly, no need to strip bucket name
        parsed = urlparse(prefix)
        if parsed.scheme == 's3':
            prefix = parsed.path.lstrip('/')
        keys = [k for k in mock_storage.keys() if k.startswith(prefix)]
        if max_keys is not None:
            keys = keys[:max_keys]

        # Strip the prefix from each key
        stripped_keys = [k[len(prefix):].lstrip("/") if k.startswith(prefix) else k for k in keys]

        if parsed.scheme == 's3':
            # Wrap keys in the expected structure
            object_info_list = [MockObjectInfo(k) for k in stripped_keys]
            return [MockListObjectsResult(object_info_list)]

        return stripped_keys

    mock_client.put_object = mock_put_object
    mock_client.get_object = mock_get_object
    mock_client.list_objects = mock_list_objects

    s3_overrides = [
        f"++workload.storage.storage_type={storage_type}",
        f"++workload.storage.storage_root={storage_root}",
        f"++workload.dataset.data_folder=s3://{storage_root}",
        "++workload.storage.storage_options.access_key_id=test-access-key",
        "++workload.storage.storage_options.secret_access_key=test-secret-key",
        "++workload.storage.storage_options.endpoint_url=https://localhost:9000",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0"
    ]

    comm.Barrier()
    yield storage_root, storage_type, mock_client, s3_overrides
    comm.Barrier()

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("fmt, framework", [("npy", "pytorch"), ("npz", "pytorch")])
def test_s3_gen_data(setup_test_env, fmt, framework) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env

    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if (comm.rank == 0):
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for generating {fmt} dataset")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=s3_overrides + [f'++workload.framework={framework}',
                                                           f'++workload.reader.data_loader={framework}',
                                                           '++workload.workflow.train=False',
                                                           '++workload.workflow.generate_data=True',
                                                           f"++workload.dataset.format={fmt}", 
                                                           "++workload.dataset.num_files_train=8", 
                                                           "++workload.dataset.num_files_eval=8"])
            benchmark = run_benchmark(cfg, verify=False)

            # Extract bucket and prefix from data_folder
            fmt = cfg.workload.dataset.format
            bucket_name = cfg.workload.storage.storage_root

            # Filter keys based on actual prefix
            train_keys = [k for k in mock_client.list_objects(bucket_name, "train/") if k.endswith(f".{fmt}")]
            valid_keys = [k for k in mock_client.list_objects(bucket_name, "valid/") if k.endswith(f".{fmt}")]
            assert len(train_keys) == cfg.workload.dataset.num_files_train
            assert len(valid_keys) == cfg.workload.dataset.num_files_eval
        
            # Clean up mock S3 after test
            clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(60, method="thread")
def test_s3_subset(setup_test_env) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO training test for subset")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            set_dftracer_finalize(False)
            # Generate data
            cfg = compose(config_name='config', overrides=s3_overrides + [
                '++workload.workflow.train=False',
                '++workload.workflow.generate_data=True'])
            benchmark = run_benchmark(cfg, verify=False)

            # Train on subset
            set_dftracer_initialize(False)
            cfg = compose(config_name='config', overrides=s3_overrides + [
                '++workload.workflow.train=True',
                '++workload.workflow.generate_data=False',
                '++workload.dataset.num_files_train=8',
                '++workload.train.computation_time=0.01'])
            benchmark = run_benchmark(cfg, verify=True)
            bucket_name = cfg.workload.storage.storage_root

        # Clean up mock S3
        clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(60, method="thread")
def test_s3_eval(setup_test_env) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if (comm.rank == 0):
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for evaluation")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                          overrides=s3_overrides + ['++workload.workflow.train=True', \
                                     '++workload.workflow.generate_data=True', \
                                     'workload.train.computation_time=0.01', \
                                     'workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=4', 
                                     '++workload.workflow.evaluation=True'])
            benchmark = run_benchmark(cfg)
            bucket_name = cfg.workload.storage.storage_root
            # Clean up mock S3 after test
            clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework, nt", [("pytorch", 0), ("pytorch", 1), ("pytorch", 2)])
def test_s3_multi_threads(setup_test_env, framework, nt) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if (comm.rank == 0):
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for generating multithreading read_threads={nt} {framework} framework")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=s3_overrides + ['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=True',
                                                           f"++workload.framework={framework}",
                                                           f"++workload.reader.data_loader={framework}",
                                                           f"++workload.reader.read_threads={nt}",
                                                           'workload.train.computation_time=0.01',
                                                           'workload.evaluation.eval_time=0.005',
                                                           '++workload.train.epochs=1',
                                                           '++workload.dataset.num_files_train=8',
                                                           '++workload.dataset.num_files_eval=8'])
            benchmark = run_benchmark(cfg)
            bucket_name = cfg.workload.storage.storage_root
        # Clean up mock S3 after test
        clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("nt, context", [(0, None), (1, "fork"), (2, "spawn"), (2, "forkserver")])
def test_s3_pytorch_multiprocessing_context(setup_test_env, nt, context, monkeypatch) -> None:
    if nt == 2 and context in ("spawn", "forkserver"):
        pytest.skip("Skipping multiprocessing test with mock client under spawn/forkserver due to patching limitations.")

    storage_root, storage_type, mock_client, s3_overrides = setup_test_env

    # Create a multiprocessing-safe mock client for this test only
    mock_storage = mock_client.storage if hasattr(mock_client, "storage") else {}
    safe_mock_client = SafeMockS3Client(mock_storage)

    # Patch globally using monkeypatch
    monkeypatch.setattr("s3torchconnector._s3client._s3client.S3Client", lambda *args, **kwargs: safe_mock_client)
    monkeypatch.setattr("dlio_benchmark.storage.s3_torch_storage.S3Client", lambda *args, **kwargs: safe_mock_client)

    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for pytorch multiprocessing_context={context} read_threads={nt}")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=s3_overrides + ['++workload.workflow.train=True',
                                                       '++workload.workflow.generate_data=True',
                                                       f"++workload.framework=pytorch",
                                                       f"++workload.reader.data_loader=pytorch",
                                                       f"++workload.reader.read_threads={nt}",
                                                       f"++workload.reader.multiprocessing_context={context}",
                                                       'workload.train.computation_time=0.01',
                                                       'workload.evaluation.eval_time=0.005',
                                                       '++workload.train.epochs=1',
                                                       '++workload.dataset.num_files_train=8',
                                                       '++workload.dataset.num_files_eval=8'])
        benchmark = run_benchmark(cfg)
        bucket_name = cfg.workload.storage.storage_root
    # Clean up mock S3 after test
    clean_s3(mock_client, bucket_name, ["train/", "valid/"])
    finalize()

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("fmt, framework, dataloader, is_even", [
                                            ("npz", "pytorch", "pytorch", True),
                                            ("npz", "pytorch", "pytorch", False),
                                            ("npy", "pytorch", "pytorch", True),
                                            ("npy", "pytorch", "pytorch", False),
                                            ])
def test_s3_train(setup_test_env, fmt, framework, dataloader, is_even) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    if is_even:
        num_files = 16
    else:
        num_files = 17
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO training test: Generating data for {fmt} format")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=s3_overrides + ['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=True',
                                                           f"++workload.framework={framework}", \
                                                           f"++workload.reader.data_loader={dataloader}", \
                                                           f"++workload.dataset.format={fmt}",
                                                           'workload.train.computation_time=0.01', \
                                                           'workload.evaluation.eval_time=0.005', \
                                                           '++workload.train.epochs=1', \
                                                           f'++workload.dataset.num_files_train={num_files}', \
                                                           '++workload.reader.read_threads=1'])
            benchmark = run_benchmark(cfg)
            bucket_name = cfg.workload.storage.storage_root
        # Clean up mock S3 after test
        clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

if __name__ == '__main__':
    unittest.main()
