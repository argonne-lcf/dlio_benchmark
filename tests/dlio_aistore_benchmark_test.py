"""
   Copyright (c) 2025, UChicago Argonne, LLC
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
from hydra import initialize_config_dir, compose
import unittest
from datetime import datetime
import uuid
import glob
from mpi4py import MPI
from tests.utils import TEST_TIMEOUT_SECONDS

comm = MPI.COMM_WORLD

import pytest
import time
import logging
import os
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
from dlio_benchmark.common.enumerations import MPIState
import dlio_benchmark

from unittest.mock import patch

config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_aistore_benchmark_test.log", mode="a", encoding='utf-8'),
        logging.StreamHandler()
    ], format='[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
)

from dlio_benchmark.main import DLIOBenchmark


# ---------------------------------------------------------------------------
# Mock classes for AIStore SDK
# ---------------------------------------------------------------------------
# These mocks replicate the AIStore SDK surface used by aistore_storage.py
# so tests can run without a real AIStore cluster or the SDK installed.


class MockAISError(Exception):
    """Mock replacement for aistore.sdk.errors.AISError"""
    def __init__(self, status=0, message=""):
        self.status = status
        super().__init__(message)


class MockAISEntry:
    """Mock replacement for list_objects_iter entries"""
    def __init__(self, name):
        self.name = name


class MockAISWriter:
    """Mock replacement for obj.get_writer()"""
    def __init__(self, key, storage):
        self.key = key
        self.storage = storage

    def put_content(self, body):
        self.storage[self.key] = body


class MockAISReader:
    """Mock replacement for obj.get_reader()"""
    def __init__(self, key, storage, byte_range=None):
        self.key = key
        self.storage = storage
        self.byte_range = byte_range

    def read_all(self):
        data = self.storage.get(self.key, b"")
        if self.byte_range:
            range_str = self.byte_range.replace("bytes=", "")
            if range_str.startswith("-"):
                # bytes=-N  ->  last N bytes
                n = int(range_str[1:])
                return data[-n:]
            elif range_str.endswith("-"):
                # bytes=N-  ->  from N to end
                start = int(range_str[:-1])
                return data[start:]
            else:
                # bytes=start-end
                parts = range_str.split("-")
                start = int(parts[0])
                end = int(parts[1])
                return data[start:end + 1]
        return data


class MockAISObject:
    """Mock replacement for bucket.object(key)"""
    def __init__(self, key, storage):
        self.key = key
        self.storage = storage

    def get_writer(self):
        return MockAISWriter(self.key, self.storage)

    def get_reader(self, byte_range=None):
        return MockAISReader(self.key, self.storage, byte_range)

    def head(self):
        if self.key in self.storage:
            return {"size": len(self.storage[self.key])}
        raise MockAISError(404, f"Object not found: {self.key}")

    def delete(self):
        self.storage.pop(self.key, None)


class MockAISBucket:
    """Mock replacement for client.bucket(name)"""
    def __init__(self, name, storage):
        self.name = name
        self.storage = storage

    def create(self, exist_ok=False):
        return self

    def head(self):
        return True

    def object(self, key):
        return MockAISObject(key, self.storage)

    def list_objects_iter(self, prefix=""):
        for key in list(self.storage.keys()):
            if key.startswith(prefix):
                yield MockAISEntry(key)


class MockAISClient:
    """Mock replacement for aistore.sdk.Client"""
    def __init__(self, storage, endpoint=None):
        self.storage = storage
        self.endpoint = endpoint

    def bucket(self, name):
        return MockAISBucket(name, self.storage)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def finalize():
    pass


def clean_aistore(mock_client, prefixes):
    """Remove keys matching any of the given prefixes from mock storage."""
    comm.Barrier()
    if comm.rank == 0:
        for prefix in prefixes:
            keys = [k for k in list(mock_client.storage.keys()) if k.startswith(prefix)]
            for key in keys:
                mock_client.storage.pop(key, None)
    comm.Barrier()


def run_benchmark(cfg, verify=True):
    comm.Barrier()
    t0 = time.time()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg["workload"])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    t1 = time.time()
    if comm.rank == 0:
        logging.info("Time for the benchmark: %.10f" % (t1 - t0))
        if verify:
            assert len(glob.glob(benchmark.output_folder + "./*_output.json")) == benchmark.comm_size
    return benchmark


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_aistore_env():
    # TorchDataset.worker_init() unpickles ConfigArguments in the main process
    # when num_workers=0, which calls DLIOMPI.reset() + set_parent_values(),
    # corrupting the singleton to CHILD_INITIALIZED state.  Reset it here so
    # subsequent tests can re-initialize properly.
    if DLIOMPI.get_instance().mpi_state == MPIState.CHILD_INITIALIZED:
        DLIOMPI.reset()
    DLIOMPI.get_instance().initialize()

    if comm.rank == 0:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        storage_root = f"ais-test-{now}-{str(uuid.uuid4())}"
    else:
        storage_root = None

    storage_root = comm.bcast(storage_root, root=0)

    # Shared in-memory mock storage
    if comm.rank == 0:
        mock_storage = {}
    else:
        mock_storage = None
    mock_storage = comm.bcast(mock_storage, root=0)

    mock_client = MockAISClient(mock_storage)

    ais_overrides = [
        "++workload.storage.storage_type=aistore",
        f"++workload.storage.storage_root={storage_root}",
        f"++workload.dataset.data_folder=s3://{storage_root}",
        "++workload.storage.storage_options.endpoint_url=http://localhost:8080",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
    ]

    with patch("dlio_benchmark.storage.aistore_storage.Client", return_value=mock_client), \
         patch("dlio_benchmark.storage.aistore_storage.AISTORE_AVAILABLE", True), \
         patch("dlio_benchmark.storage.aistore_storage.AISError", MockAISError), \
         patch("dlio_benchmark.storage.storage_factory.AISTORE_AVAILABLE", True):
        comm.Barrier()
        yield storage_root, mock_client, ais_overrides
        comm.Barrier()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework", [("npy", "pytorch"), ("npz", "pytorch")])
def test_aistore_gen_data(setup_aistore_env, fmt, framework):
    storage_root, mock_client, ais_overrides = setup_aistore_env

    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO AIStore test for generating {fmt} dataset")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=ais_overrides + [
            f'++workload.framework={framework}',
            f'++workload.reader.data_loader={framework}',
            '++workload.workflow.train=False',
            '++workload.workflow.generate_data=True',
            f'++workload.dataset.format={fmt}',
            '++workload.dataset.num_files_train=8',
            '++workload.dataset.num_files_eval=8',
        ])
        benchmark = run_benchmark(cfg, verify=False)

        # Count generated files in mock storage
        fmt_ext = cfg.workload.dataset.format
        train_keys = [k for k in mock_client.storage.keys()
                      if k.startswith("train/") and k.endswith(f".{fmt_ext}")]
        valid_keys = [k for k in mock_client.storage.keys()
                      if k.startswith("valid/") and k.endswith(f".{fmt_ext}")]
        assert len(train_keys) == cfg.workload.dataset.num_files_train
        assert len(valid_keys) == cfg.workload.dataset.num_files_eval

        clean_aistore(mock_client, ["train/", "valid/"])
    finalize()


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework, is_even", [
    ("npy", "pytorch", True),
    ("npy", "pytorch", False),
    ("npz", "pytorch", True),
    ("npz", "pytorch", False),
])
def test_aistore_train(setup_aistore_env, fmt, framework, is_even):
    storage_root, mock_client, ais_overrides = setup_aistore_env
    num_files = 16 if is_even else 17

    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO AIStore training test: {fmt} format, num_files={num_files}")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=ais_overrides + [
            '++workload.workflow.train=True',
            '++workload.workflow.generate_data=True',
            f'++workload.framework={framework}',
            f'++workload.reader.data_loader={framework}',
            f'++workload.dataset.format={fmt}',
            'workload.train.computation_time=0.01',
            'workload.evaluation.eval_time=0.005',
            '++workload.train.epochs=1',
            f'++workload.dataset.num_files_train={num_files}',
            '++workload.reader.read_threads=1',
        ])
        benchmark = run_benchmark(cfg)
        clean_aistore(mock_client, ["train/", "valid/"])
    finalize()


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_aistore_eval(setup_aistore_env):
    storage_root, mock_client, ais_overrides = setup_aistore_env

    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(" DLIO AIStore test for evaluation")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=ais_overrides + [
            '++workload.workflow.train=True',
            '++workload.workflow.generate_data=True',
            'workload.train.computation_time=0.01',
            'workload.evaluation.eval_time=0.005',
            '++workload.train.epochs=4',
            '++workload.workflow.evaluation=True',
        ])
        benchmark = run_benchmark(cfg)
        clean_aistore(mock_client, ["train/", "valid/"])
    finalize()


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, nt", [("pytorch", 0), ("pytorch", 1), ("pytorch", 2)])
def test_aistore_multi_threads(setup_aistore_env, framework, nt):
    storage_root, mock_client, ais_overrides = setup_aistore_env

    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO AIStore test for multithreading read_threads={nt} {framework}")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=ais_overrides + [
            '++workload.workflow.train=True',
            '++workload.workflow.generate_data=True',
            f'++workload.framework={framework}',
            f'++workload.reader.data_loader={framework}',
            f'++workload.reader.read_threads={nt}',
            'workload.train.computation_time=0.01',
            'workload.evaluation.eval_time=0.005',
            '++workload.train.epochs=1',
            '++workload.dataset.num_files_train=8',
            '++workload.dataset.num_files_eval=8',
        ])
        benchmark = run_benchmark(cfg)
        clean_aistore(mock_client, ["train/", "valid/"])
    finalize()


if __name__ == '__main__':
    unittest.main()
