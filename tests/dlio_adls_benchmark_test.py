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
from hydra import initialize, initialize_config_dir, compose
from omegaconf import OmegaConf
import unittest
from datetime import datetime, timedelta, timezone
import uuid
import io
from io import BytesIO
import glob
from mpi4py import MPI
from tests.utils import TEST_TIMEOUT_SECONDS

comm = MPI.COMM_WORLD

import pytest
import time
import subprocess
import logging
import os
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
import dlio_benchmark

from unittest.mock import patch, MagicMock
try:
    from azstoragetorch.io import BlobIO
except ImportError as e:
    BlobIO = None
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

def clean_adls(mock_file_system_client, prefixes: list[str]) -> None:
    """Clean up mock ADLS Gen2 storage after tests"""
    comm.Barrier()
    if comm.rank == 0:
        for prefix in prefixes:
            # Get all paths starting with prefix
            keys = [k for k in mock_file_system_client.storage.keys() if k.startswith(prefix)]
            for key in keys:
                del mock_file_system_client.storage[key]
    comm.Barrier()

def get_adls_prefixes_from_uri(uri: str, subdirs=("train", "valid")):
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

class MockADLSFileClient:
    """Mock Azure Data Lake Storage Gen2 file client"""
    def __init__(self, file_system_client, file_path):
        self.file_system_client = file_system_client
        self.file_path = file_path
        self.storage = file_system_client.storage
    
    def create_file(self):
        """Create a file (no-op for mock)"""
        pass
    
    def upload_data(self, data, overwrite=True):
        """Upload data to the file"""
        if isinstance(data, bytes):
            self.storage[self.file_path] = data
        elif isinstance(data, str):
            self.storage[self.file_path] = data.encode('utf-8')
        else:
            self.storage[self.file_path] = bytes(data)
    
    def download_file(self, offset=None, length=None):
        """Download file data"""
        data = self.storage.get(self.file_path, b"")
        if offset is not None and length is not None:
            return MockDownloadStream(data[offset:offset+length])
        return MockDownloadStream(data)
    
    def delete_file(self):
        """Delete a file"""
        if self.file_path in self.storage:
            del self.storage[self.file_path]
    
    def get_file_properties(self):
        """Get file properties"""
        if self.file_path in self.storage:
            return {'is_directory': False}
        raise Exception(f"File not found: {self.file_path}")

class MockDownloadStream:
    """Mock download stream"""
    def __init__(self, data):
        self.data = data
    
    def readall(self):
        return self.data

class MockADLSDirectoryClient:
    """Mock Azure Data Lake Storage Gen2 directory client"""
    def __init__(self, file_system_client, directory_path):
        self.file_system_client = file_system_client
        self.directory_path = directory_path
        self.storage = file_system_client.storage
    
    def create_directory(self):
        """Create a directory (mark in storage)"""
        # Store directory marker
        if not self.directory_path.endswith('/'):
            dir_key = self.directory_path + '/'
        else:
            dir_key = self.directory_path
        self.storage[dir_key] = b""
    
    def delete_directory(self):
        """Delete a directory and all its contents"""
        prefix = self.directory_path if self.directory_path.endswith('/') else self.directory_path + '/'
        keys_to_delete = [k for k in self.storage.keys() if k.startswith(prefix)]
        for key in keys_to_delete:
            del self.storage[key]
    
    def get_directory_properties(self):
        """Get directory properties"""
        return {'is_directory': True}

class MockPathItem:
    """Mock path item returned by get_paths"""
    def __init__(self, name, is_directory=False):
        self.name = name
        self.is_directory = is_directory

class MockADLSFileSystemClient:
    """Mock Azure Data Lake Storage Gen2 file system client"""
    def __init__(self, file_system_name):
        self.file_system_name = file_system_name
        self.storage = {}
    
    def create_file_system(self):
        """Create file system (no-op for mock)"""
        pass
    
    def get_file_system_properties(self):
        """Get file system properties"""
        return {'name': self.file_system_name}
    
    def get_file_client(self, file_path):
        """Get a file client"""
        return MockADLSFileClient(self, file_path)
    
    def get_directory_client(self, directory_path):
        """Get a directory client"""
        return MockADLSDirectoryClient(self, directory_path)
    
    def get_paths(self, path="", recursive=True):
        """List paths under a given path"""
        prefix = path if path.endswith('/') or path == "" else path + '/'
        if path == "":
            # List all items
            paths = []
            seen = set()
            for key in self.storage.keys():
                if key:  # Skip empty keys
                    # Get the top-level name
                    first_part = key.split('/')[0]
                    if first_part not in seen:
                        seen.add(first_part)
                        is_dir = '/' in key[len(first_part):]
                        # Return full path from root (matching Azure SDK behavior)
                        paths.append(MockPathItem(first_part, is_directory=is_dir))
            return paths
        else:
            # List items under specific path
            paths = []
            seen = set()
            for key in self.storage.keys():
                if key.startswith(prefix):
                    # Get relative path
                    relative = key[len(prefix):]
                    if relative:
                        # Get first component
                        first_part = relative.split('/')[0]
                        if first_part and first_part not in seen:
                            seen.add(first_part)
                            # Check if this is a directory
                            full_path = prefix + first_part
                            is_dir = any(k.startswith(full_path + '/') for k in self.storage.keys())
                            # Return full path from root (matching Azure SDK behavior)
                            paths.append(MockPathItem(full_path, is_directory=is_dir))
            return paths

class MockDataLakeServiceClient:
    """Mock Azure Data Lake Service Client"""
    def __init__(self, account_url=None, credential=None):
        self.account_url = account_url
        self.credential = credential
        self._file_systems = {}
    
    @classmethod
    def from_connection_string(cls, connection_string):
        """Create from connection string"""
        return cls(account_url="mock_url")
    
    def get_file_system_client(self, file_system):
        """Get or create a file system client"""
        if file_system not in self._file_systems:
            self._file_systems[file_system] = MockADLSFileSystemClient(file_system)
        return self._file_systems[file_system]

@pytest.fixture
def setup_test_env():
    DLIOMPI.get_instance().initialize()
    if comm.rank == 0:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        storage_root = f"adls-test-container-{now}-{str(uuid.uuid4())}"
        storage_type = "adls_gen2"
    else:
        storage_root = None
        storage_type = None

    storage_root = comm.bcast(storage_root, root=0)
    storage_type = comm.bcast(storage_type, root=0)

    # Create mock ADLS Gen2 service client
    if comm.rank == 0:
        mock_service_client = MockDataLakeServiceClient()
        mock_file_system_client = mock_service_client.get_file_system_client(storage_root)
        mock_file_system_client.create_file_system()
        # Initialize with a marker file
        mock_file_system_client.storage["init.txt"] = b"container initialized"
        mock_storage = mock_file_system_client.storage
    else:
        mock_storage = None
        mock_service_client = MockDataLakeServiceClient()
        mock_file_system_client = mock_service_client.get_file_system_client(storage_root)

    # Broadcast the mock_storage dictionary to all ranks
    mock_storage = comm.bcast(mock_storage, root=0)
    mock_file_system_client.storage = mock_storage

    adls_overrides = [
        f"++workload.storage.storage_type={storage_type}",
        f"++workload.storage.storage_root={storage_root}",
        f"++workload.dataset.data_folder=abfs://{storage_root}@test.dfs.core.windows.net",
        "++workload.storage.storage_options.account_name=test",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0"
    ]

    comm.Barrier()
    yield storage_root, storage_type, mock_file_system_client, adls_overrides
    comm.Barrier()

@pytest.fixture
def patch_adls_checkpoint(setup_test_env):
    storage_root, storage_type, mock_file_system_client, adls_overrides = setup_test_env
    adls_overrides += [f"++workload.checkpoint.checkpoint_folder=abfs://{storage_root}@test.dfs.core.windows.net/checkpoints"]
    checkpoint_storage = {}

    from dlio_benchmark.checkpointing.pytorch_adls_checkpointing import PyTorchADLSCheckpointing
    PyTorchADLSCheckpointing._PyTorchADLSCheckpointing__instance = None
    
    class MockBlobIO:
        """Mock BlobIO for testing"""
        def __init__(self, blob_url, mode, credential=None, **kwargs):
            self.blob_url = blob_url
            self.mode = mode
            self.credential = credential
            self._mock_storage = checkpoint_storage
            self._buffer = None

        def __enter__(self):
            if self.mode == "wb":
                self._buffer = io.BytesIO()
                return self._buffer
            data = self._mock_storage.get(self.blob_url, b'')
            return io.BytesIO(data)

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None and self.mode == "wb" and self._buffer is not None:
                self._mock_storage[self.blob_url] = self._buffer.getvalue()
            return False
    
    # Always mock BlobIO for tests (whether azstoragetorch is installed or not)
    with patch("dlio_benchmark.checkpointing.pytorch_adls_checkpointing.BlobIO", MockBlobIO):
        mock_file_system_client.checkpoint_storage = checkpoint_storage
        yield setup_test_env

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework", [("npy", "pytorch"), ("npz", "pytorch")])
def test_adls_gen_data(setup_test_env, fmt, framework) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = setup_test_env

    # Patch both DataLakeServiceClient and DefaultAzureCredential
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if (comm.rank == 0):
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for generating {fmt} dataset on ADLS Gen2")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=adls_overrides + [f'++workload.framework={framework}',
                                                           f'++workload.reader.data_loader={framework}',
                                                           '++workload.workflow.train=False',
                                                           '++workload.workflow.generate_data=True',
                                                           f"++workload.dataset.format={fmt}", 
                                                           "++workload.dataset.num_files_train=8", 
                                                           "++workload.dataset.num_files_eval=8"])
            benchmark = run_benchmark(cfg, verify=False)

            # Verify files were created
            train_keys = [k for k in mock_file_system_client.storage.keys() if k.startswith("train/") and k.endswith(f".{fmt}")]
            valid_keys = [k for k in mock_file_system_client.storage.keys() if k.startswith("valid/") and k.endswith(f".{fmt}")]
            assert len(train_keys) == cfg.workload.dataset.num_files_train, f"Expected {cfg.workload.dataset.num_files_train} train files, got {len(train_keys)}"
            assert len(valid_keys) == cfg.workload.dataset.num_files_eval, f"Expected {cfg.workload.dataset.num_files_eval} valid files, got {len(valid_keys)}"
        
            # Clean up mock ADLS after test
            clean_adls(mock_file_system_client, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_adls_subset(setup_test_env) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = setup_test_env
    
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO training test for subset on ADLS Gen2")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            set_dftracer_finalize(False)
            # Generate data
            cfg = compose(config_name='config', overrides=adls_overrides + [
                '++workload.workflow.train=False',
                '++workload.workflow.generate_data=True'])
            benchmark = run_benchmark(cfg, verify=False)

            # Train on subset
            set_dftracer_initialize(False)
            cfg = compose(config_name='config', overrides=adls_overrides + [
                '++workload.workflow.train=True',
                '++workload.workflow.generate_data=False',
                '++workload.dataset.num_files_train=8',
                '++workload.train.computation_time=0.01'])
            benchmark = run_benchmark(cfg, verify=True)
            
        # Clean up
        clean_adls(mock_file_system_client, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_adls_eval(setup_test_env) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = setup_test_env
    
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO evaluation test on ADLS Gen2")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=adls_overrides + ['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=True',
                                                           'workload.train.computation_time=0.01',
                                                           'workload.evaluation.eval_time=0.005',
                                                           '++workload.train.epochs=4',
                                                           '++workload.workflow.evaluation=True'])
            benchmark = run_benchmark(cfg)
            
        # Clean up
        clean_adls(mock_file_system_client, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, nt", [("pytorch", 0), ("pytorch", 1), ("pytorch", 2)])
def test_adls_multi_threads(setup_test_env, framework, nt) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = setup_test_env
    
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO multi-threaded test on ADLS Gen2: {framework} with {nt} threads")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=adls_overrides + ['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=True',
                                                           f'++workload.framework={framework}',
                                                           f'++workload.reader.data_loader={framework}',
                                                           '++workload.dataset.format=npz',
                                                           '++workload.train.computation_time=0.01',
                                                           '++workload.evaluation.eval_time=0.005',
                                                           '++workload.train.epochs=1',
                                                           '++workload.dataset.num_files_train=8',
                                                           f'++workload.reader.read_threads={nt}'])
            benchmark = run_benchmark(cfg)
            
        # Clean up
        clean_adls(mock_file_system_client, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("nt, context", [(0, None), (1, "fork")])
def test_adls_pytorch_multiprocessing_context(setup_test_env, nt, context, monkeypatch) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = setup_test_env
    
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO PyTorch multiprocessing context test on ADLS Gen2: threads={nt}, context={context}")
            logging.info("=" * 80)
        
        overrides = adls_overrides + ['++workload.workflow.train=True',
                                      '++workload.workflow.generate_data=True',
                                      '++workload.framework=pytorch',
                                      '++workload.reader.data_loader=pytorch',
                                      '++workload.dataset.format=npz',
                                      '++workload.train.computation_time=0.01',
                                      '++workload.evaluation.eval_time=0.005',
                                      '++workload.train.epochs=1',
                                      '++workload.dataset.num_files_train=8',
                                      f'++workload.reader.read_threads={nt}']
        
        if context is not None:
            overrides.append(f'++workload.reader.multiprocessing_context={context}')
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=overrides)
            benchmark = run_benchmark(cfg)
            
        # Clean up
        clean_adls(mock_file_system_client, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework, dataloader, is_even", [
    ("npy", "pytorch", "pytorch", True),
    ("npz", "pytorch", "pytorch", True),
    ("npy", "pytorch", "pytorch", False),
    ("npz", "pytorch", "pytorch", False)
])
def test_adls_train(setup_test_env, fmt, framework, dataloader, is_even) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = setup_test_env
    if is_even:
        num_files = 16
    else:
        num_files = 17
    
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO training test on ADLS Gen2: Generating data for {fmt} format")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=adls_overrides + ['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=True',
                                                           f"++workload.framework={framework}",
                                                           f"++workload.reader.data_loader={dataloader}",
                                                           f"++workload.dataset.format={fmt}",
                                                           'workload.train.computation_time=0.01',
                                                           'workload.evaluation.eval_time=0.005',
                                                           '++workload.train.epochs=1',
                                                           f'++workload.dataset.num_files_train={num_files}',
                                                           '++workload.reader.read_threads=1'])
            benchmark = run_benchmark(cfg)
            
        # Clean up
        clean_adls(mock_file_system_client, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, model_size, optimizers, num_layers, layer_params, zero_stage, randomize", [
    ("pytorch", 1024, [1024, 128], 2, [16], 0, True),
    ("pytorch", 1024, [1024, 128], 2, [16], 3, True),
    ("pytorch", 1024, [128], 1, [16], 0, True),
    ("pytorch", 1024, [1024, 128], 2, [16], 0, False),
    ("pytorch", 1024, [1024, 128], 2, [16], 3, False),
    ("pytorch", 1024, [128], 1, [16], 0, False)
])
def test_adls_checkpoint_epoch(patch_adls_checkpoint, framework, model_size, optimizers, num_layers, layer_params, zero_stage, randomize) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = patch_adls_checkpoint
    
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for checkpointing at the end of epochs on ADLS Gen2")
            logging.info("=" * 80)
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            epochs = 8
            epoch_per_ckp = 2
            cfg = compose(config_name='config',
                         overrides=adls_overrides + [
                             f'++workload.framework={framework}',
                             f'++workload.reader.data_loader={framework}',
                             '++workload.workflow.train=True',
                             '++workload.workflow.generate_data=True',
                             f'++workload.checkpoint.randomize_tensor={randomize}',
                             '++workload.train.computation_time=0.01',
                             '++workload.evaluation.eval_time=0.005',
                             f'++workload.train.epochs={epochs}',
                             '++workload.workflow.checkpoint=True',
                             '++workload.checkpoint.checkpoint_mechanism=pt_adls_save',
                             f'++workload.checkpoint.epochs_between_checkpoints={epoch_per_ckp}',
                             f'++workload.model.model_size={model_size}',
                             f'++workload.model.optimization_groups={optimizers}',
                             f'++workload.model.num_layers={num_layers}',
                             f'++workload.model.parallelism.zero_stage={zero_stage}',
                             f'++workload.model.layer_parameters={layer_params}',
                             f'++workload.model.parallelism.tensor={comm.size}'
                         ])
            ConfigArguments.reset()
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()
            benchmark.run()
            benchmark.finalize()

            checkpoint_keys = list(mock_file_system_client.checkpoint_storage.keys())
            n = 0
            if len(layer_params) > 0:
                n = num_layers
            nranks = comm.size
            num_model_files = 1
            num_optimizer_files = 1
            num_layer_files = 1
            files_per_checkpoint = (num_model_files + num_optimizer_files + num_layer_files) * nranks
            if framework == "pytorch":
                num_check_files = epochs / epoch_per_ckp * files_per_checkpoint
                assert (len(checkpoint_keys) == num_check_files), f"files produced are {len(checkpoint_keys)} {num_check_files} {checkpoint_keys}"
        
        # Clean up
        clean_adls(mock_file_system_client, ["checkpoints/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_adls_checkpoint_step(patch_adls_checkpoint) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = patch_adls_checkpoint
    
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for checkpointing at the end of steps on ADLS Gen2")
            logging.info("=" * 80)
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                         overrides=adls_overrides + [
                             '++workload.workflow.train=True',
                             '++workload.workflow.generate_data=True',
                             '++workload.train.computation_time=0.01',
                             '++workload.evaluation.eval_time=0.005',
                             '++workload.train.epochs=8',
                             '++workload.workflow.checkpoint=True',
                             '++workload.checkpoint.checkpoint_mechanism=pt_adls_save',
                             '++workload.checkpoint.steps_between_checkpoints=2'
                         ])
            ConfigArguments.reset()
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()
            benchmark.run()
            benchmark.finalize()

            dataset = cfg['workload']['dataset']
            nstep = dataset.num_files_train * dataset.num_samples_per_file // cfg['workload']['reader'].batch_size // benchmark.comm_size
            ncheckpoints = nstep // 2 * 8
            checkpoint_keys = list(mock_file_system_client.checkpoint_storage.keys())
            assert (len(checkpoint_keys) == ncheckpoints)
        
        # Clean up
        clean_adls(mock_file_system_client, ["checkpoints/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_adls_checkpoint_ksm_config(patch_adls_checkpoint) -> None:
    storage_root, storage_type, mock_file_system_client, adls_overrides = patch_adls_checkpoint
    
    with patch("dlio_benchmark.storage.adls_gen2_storage.DataLakeServiceClient") as mock_service, \
         patch("dlio_benchmark.storage.adls_gen2_storage.DefaultAzureCredential") as mock_cred:
        mock_instance = MagicMock()
        mock_instance.get_file_system_client.return_value = mock_file_system_client
        mock_service.return_value = mock_instance
        mock_service.from_connection_string.return_value = mock_instance
        mock_cred.return_value = MagicMock()

        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(" DLIO test for KSM config on ADLS Gen2")
            logging.info("=" * 80)
        
        # Test Case 1: KSM enabled with defaults
        logging.info("Testing KSM enabled with defaults...")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                         overrides=adls_overrides + [
                             '++workload.workflow.checkpoint=True',
                             '++workload.checkpoint.checkpoint_mechanism=pt_adls_save',
                             '++workload.checkpoint.ksm={}',
                             '++workload.workflow.generate_data=False',
                             '++workload.workflow.train=False',
                             '++workload.checkpoint.num_checkpoints_write=1',
                             '++workload.checkpoint.num_checkpoints_read=1',
                             '++workload.checkpoint.randomize_tensor=False'
                         ])
            ConfigArguments.reset()
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()
            
            args = ConfigArguments.get_instance()
            assert args.ksm_init is True, "[Test Case 1 Failed] ksm_init should be True when ksm section is present"
            assert args.ksm_madv_mergeable_id == 12
            assert args.ksm_high_ram_trigger == 30.0
            assert args.ksm_low_ram_exit == 15.0
            assert args.ksm_await_time == 200
            logging.info("[Test Case 1 Passed]")
        
        # Test Case 2: KSM enabled with overrides
        logging.info("Testing KSM enabled with overrides...")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                         overrides=adls_overrides + [
                             '++workload.workflow.checkpoint=True',
                             '++workload.checkpoint.checkpoint_mechanism=pt_adls_save',
                             '++workload.checkpoint.ksm.high_ram_trigger=25.5',
                             '++workload.checkpoint.ksm.await_time=100',
                             '++workload.workflow.generate_data=False',
                             '++workload.workflow.train=False',
                             '++workload.checkpoint.num_checkpoints_write=1',
                             '++workload.checkpoint.num_checkpoints_read=1',
                             '++workload.checkpoint.randomize_tensor=False'
                         ])
            ConfigArguments.reset()
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()
            
            args = ConfigArguments.get_instance()
            assert args.ksm_init is True
            assert args.ksm_high_ram_trigger == 25.5
            assert args.ksm_await_time == 100
            logging.info("[Test Case 2 Passed]")
        
        # Test Case 3: KSM disabled
        logging.info("Testing KSM disabled...")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                         overrides=adls_overrides + [
                             '++workload.workflow.checkpoint=True',
                             '++workload.checkpoint.checkpoint_mechanism=pt_adls_save',
                             '++workload.workflow.generate_data=False',
                             '++workload.workflow.train=False',
                             '++workload.checkpoint.num_checkpoints_write=1',
                             '++workload.checkpoint.num_checkpoints_read=1',
                             '++workload.checkpoint.randomize_tensor=False'
                         ])
            ConfigArguments.reset()
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()
            
            args = ConfigArguments.get_instance()
            assert args.ksm_init is False
            logging.info("[Test Case 3 Passed]")
        
        # Clean up
        clean_adls(mock_file_system_client, ["checkpoints/"])
        finalize()

if __name__ == '__main__':
    unittest.main()


def test_adls_checkpoint_uses_cached_container_sas(monkeypatch):
    from dlio_benchmark.checkpointing import pytorch_adls_checkpointing as module
    from dlio_benchmark.checkpointing.pytorch_adls_checkpointing import PyTorchADLSCheckpointing

    mock_generate_container_sas = MagicMock(return_value="sig=container-token")
    monkeypatch.setattr(module, "generate_container_sas", mock_generate_container_sas)

    checkpoint = PyTorchADLSCheckpointing.__new__(PyTorchADLSCheckpointing)
    checkpoint._account_name = "testacct"
    checkpoint._account_key = "testkey"
    checkpoint._shared_access_signature = None
    checkpoint._container_sas_tokens = {}
    checkpoint._container_sas_refresh_margin = timedelta(minutes=5)
    checkpoint._container_sas_ttl = timedelta(hours=1)

    first = checkpoint._to_blob_url(
        "abfs://cont@testacct.dfs.core.windows.net/checkpoints/model.pt", for_write=True
    )
    second = checkpoint._to_blob_url(
        "abfs://cont@testacct.dfs.core.windows.net/checkpoints/optim.pt", for_write=True
    )
    third = checkpoint._to_blob_url(
        "abfs://cont@testacct.dfs.core.windows.net/checkpoints/model.pt", for_write=False
    )

    assert "sig=container-token" in first
    assert "sig=container-token" in second
    assert "sig=container-token" in third
    assert mock_generate_container_sas.call_count == 1


def test_adls_checkpoint_container_sas_cached_per_container(monkeypatch):
    from dlio_benchmark.checkpointing import pytorch_adls_checkpointing as module
    from dlio_benchmark.checkpointing.pytorch_adls_checkpointing import PyTorchADLSCheckpointing

    mock_generate_container_sas = MagicMock(return_value="sig=container-token")
    monkeypatch.setattr(module, "generate_container_sas", mock_generate_container_sas)

    checkpoint = PyTorchADLSCheckpointing.__new__(PyTorchADLSCheckpointing)
    checkpoint._account_name = "testacct"
    checkpoint._account_key = "testkey"
    checkpoint._shared_access_signature = None
    checkpoint._container_sas_tokens = {}
    checkpoint._container_sas_refresh_margin = timedelta(minutes=5)
    checkpoint._container_sas_ttl = timedelta(hours=1)

    checkpoint._to_blob_url("abfs://conta@testacct.dfs.core.windows.net/checkpoints/model.pt", for_write=True)
    checkpoint._to_blob_url("abfs://contb@testacct.dfs.core.windows.net/checkpoints/model.pt", for_write=True)

    assert mock_generate_container_sas.call_count == 2


def test_adls_checkpoint_container_sas_refreshes_near_expiry(monkeypatch):
    from dlio_benchmark.checkpointing import pytorch_adls_checkpointing as module
    from dlio_benchmark.checkpointing.pytorch_adls_checkpointing import PyTorchADLSCheckpointing

    mock_generate_container_sas = MagicMock(return_value="sig=fresh-token")
    monkeypatch.setattr(module, "generate_container_sas", mock_generate_container_sas)

    checkpoint = PyTorchADLSCheckpointing.__new__(PyTorchADLSCheckpointing)
    checkpoint._account_name = "testacct"
    checkpoint._account_key = "testkey"
    checkpoint._shared_access_signature = None
    checkpoint._container_sas_refresh_margin = timedelta(minutes=5)
    checkpoint._container_sas_ttl = timedelta(hours=1)
    checkpoint._container_sas_tokens = {
        "cont": {
            "token": "sig=stale-token",
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=1),
        }
    }

    url = checkpoint._to_blob_url("abfs://cont@testacct.dfs.core.windows.net/checkpoints/model.pt", for_write=True)

    assert "sig=fresh-token" in url
    assert mock_generate_container_sas.call_count == 1
