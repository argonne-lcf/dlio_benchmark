"""
   Copyright (c) 2026, UChicago Argonne, LLC
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
import glob
import io
import logging
import os
import uuid
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from mpi4py import MPI

import dlio_benchmark
from dlio_benchmark.main import DLIOBenchmark, set_dftracer_finalize, set_dftracer_initialize
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
from tests.utils import TEST_TIMEOUT_SECONDS

pytest.importorskip("multistorageclient")

comm = MPI.COMM_WORLD

config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_benchmark_test.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    format="[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]",
)


def finalize():
    pass


def run_benchmark(cfg, verify=True):
    comm.Barrier()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg["workload"])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    if comm.rank == 0 and verify:
        assert len(glob.glob(benchmark.output_folder + "./*_output.json")) == benchmark.comm_size
    return benchmark


class MockMscClient:
    """
    Minimal MSC StorageClient stand-in: same call patterns as dlio_benchmark.storage.msc_storage.MscStorage.
    Merges the object store across MPI ranks after mutating operations so walk_node/train work under mpirun.
    """

    def __init__(self, store: dict, mpi_comm):
        self._store = store
        self._comm = mpi_comm
        self._primary_root: str | None = None

    def _merge_from_all_ranks(self):
        self._comm.Barrier()
        chunks = self._comm.allgather(dict(self._store))
        merged: dict = {}
        for d in chunks:
            merged.update(d)
        self._store.clear()
        self._store.update(merged)

    def upload_file(self, dest_uri, src):
        if isinstance(src, str):
            with open(src, "rb") as f:
                data = f.read()
        else:
            data = src.read()
        self._store[dest_uri] = data
        self._merge_from_all_ranks()

    def read_file(self, uri, byte_range=None):
        data = self._store.get(uri)
        if data is None:
            raise FileNotFoundError(uri)
        if byte_range is None:
            return data
        off = getattr(byte_range, "offset", None)
        ln = getattr(byte_range, "length", None)
        if ln is None:
            ln = getattr(byte_range, "size", None)
        if off is None:
            raise ValueError(f"Unsupported byte_range: {byte_range!r}")
        return data[off : off + int(ln)]

    def read(self, uri):
        return self.read_file(uri, byte_range=None)

    def open(self, uri):
        raw = self.read(uri)
        return BytesIO(raw)

    def delete(self, uri, recursive=True):
        if not recursive:
            self._store.pop(uri, None)
            self._merge_from_all_ranks()
            return
        prefix = uri.rstrip("/")
        to_del = [k for k in list(self._store) if k == prefix or k.startswith(prefix + "/")]
        for k in to_del:
            del self._store[k]
        self._merge_from_all_ranks()

    def info(self, uri):
        if uri in self._store:
            return SimpleNamespace(type="file")
        u = uri.rstrip("/")
        for k in self._store:
            if k.startswith(u + "/"):
                return SimpleNamespace(type="directory")
        raise FileNotFoundError(uri)

    def list(self, path):
        p = path.rstrip("/")
        pref = p + "/"
        seen = []
        for k in sorted(self._store):
            if not k.startswith(pref):
                continue
            rest = k[len(pref) :]
            if "/" in rest:
                continue
            seen.append(SimpleNamespace(key=k))
        return seen


def _make_resolve_fn(client: MockMscClient):
    def resolve_storage_client(namespace: str):
        if namespace.startswith("msc://"):
            root = namespace if namespace.endswith("/") else namespace + "/"
            client._primary_root = root
            return client, root
        base = client._primary_root or "msc://dlio-msc-unset/"
        base = base.rstrip("/")
        root = f"{base}/{namespace.strip('/')}/"
        return client, root

    return resolve_storage_client


def _msc_torch_bridge(store: dict, msc_client: MockMscClient):
    def save(state, path):
        buf = BytesIO()
        torch.save(state, buf)
        store[path] = buf.getvalue()
        msc_client._merge_from_all_ranks()

    def load(path, map_location=None):
        buf = BytesIO(store[path])
        return torch.load(buf, map_location=map_location, weights_only=False)

    return SimpleNamespace(save=save, load=load)


def _clean_msc_prefix(store: dict, prefix: str):
    comm.Barrier()
    p = prefix.rstrip("/")
    for k in list(store.keys()):
        if k.startswith(p):
            del store[k]
    comm.Barrier()


@pytest.fixture
def msc_test_env():
    DLIOMPI.get_instance().initialize()
    store: dict = {}
    client = MockMscClient(store, comm)
    resolve_fn = _make_resolve_fn(client)

    if comm.rank == 0:
        storage_root = f"msc://dlio-msc-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4()}/"
    else:
        storage_root = None
    storage_root = comm.bcast(storage_root, root=0)

    overrides = [
        "++workload.storage.storage_type=msc",
        f"++workload.storage.storage_root={storage_root}",
        "++workload.dataset.data_folder=msc_dataset",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
    ]

    import dlio_benchmark.storage.msc_storage as msc_storage_mod
    import dlio_benchmark.checkpointing.pytorch_msc_checkpointing as pt_msc_mod

    mock_torch = _msc_torch_bridge(store, client)
    with patch.object(msc_storage_mod.msc, "resolve_storage_client", side_effect=resolve_fn):
        with patch.object(pt_msc_mod.msc, "torch", mock_torch, create=True):
            comm.Barrier()
            yield storage_root, store, overrides
            comm.Barrier()


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_msc_gen_data_indexed_binary(msc_test_env):
    storage_root, store, msc_overrides = msc_test_env
    if comm.rank == 0:
        logging.info("MSC test: indexed_binary data generation")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=msc_overrides
            + [
                "++workload.framework=pytorch",
                "++workload.reader.data_loader=pytorch",
                "++workload.workflow.train=False",
                "++workload.workflow.generate_data=True",
                "++workload.dataset.format=indexed_binary",
                "++workload.dataset.num_files_train=2",
                "++workload.dataset.num_files_eval=1",
                "++workload.dataset.num_samples_per_file=4",
                "++workload.dataset.record_length_bytes=256",
            ],
        )
        run_benchmark(cfg, verify=False)

    if comm.rank == 0:
        train_prefix = os.path.join(storage_root.rstrip("/"), "msc_dataset", "train").replace("\\", "/")
        for i in range(2):
            base = f"{train_prefix}/img_{i}_of_2.indexed_binary"
            assert base in store, f"missing {base}"
            assert base + ".off.idx" in store
            assert base + ".sz.idx" in store
        valid_prefix = os.path.join(storage_root.rstrip("/"), "msc_dataset", "valid").replace("\\", "/")
        v0 = f"{valid_prefix}/img_0_of_1.indexed_binary"
        assert v0 in store
        assert v0 + ".off.idx" in store
        assert v0 + ".sz.idx" in store
    finalize()
    _clean_msc_prefix(store, storage_root)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_msc_train_indexed_binary_pytorch(msc_test_env):
    storage_root, store, msc_overrides = msc_test_env
    if comm.rank == 0:
        logging.info("MSC test: indexed_binary train (pytorch dataloader)")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        set_dftracer_finalize(False)
        cfg = compose(
            config_name="config",
            overrides=msc_overrides
            + [
                "++workload.framework=pytorch",
                "++workload.reader.data_loader=pytorch",
                "++workload.workflow.train=True",
                "++workload.workflow.generate_data=True",
                "++workload.dataset.format=indexed_binary",
                "++workload.dataset.num_files_train=2",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_samples_per_file=8",
                "++workload.dataset.record_length_bytes=512",
                "++workload.reader.read_threads=1",
                "++workload.reader.multiprocessing_context=fork",
                "++workload.train.computation_time=0.01",
                "++workload.train.epochs=1",
            ],
        )
        run_benchmark(cfg, verify=True)
    finalize()
    _clean_msc_prefix(store, storage_root)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize(
    "framework, model_size, optimizers, num_layers, layer_params, zero_stage, randomize",
    [("pytorch", 1024, [1024, 128], 2, [16], 0, True)],
)
def test_msc_checkpoint_pytorch(
    msc_test_env,
    framework,
    model_size,
    optimizers,
    num_layers,
    layer_params,
    zero_stage,
    randomize,
):
    storage_root, store, msc_overrides = msc_test_env
    if comm.rank == 0:
        logging.info("MSC test: PyTorch MSC checkpointing")

    msc_overrides = list(msc_overrides) + ["++workload.checkpoint.checkpoint_folder=checkpoints"]

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        set_dftracer_initialize(False)
        epochs = 4
        epoch_per_ckp = 2
        cfg = compose(
            config_name="config",
            overrides=msc_overrides
            + [
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=True",
                "++workload.workflow.generate_data=True",
                "++workload.workflow.checkpoint=True",
                f"++workload.checkpoint.randomize_tensor={randomize}",
                "++workload.train.computation_time=0.01",
                "++workload.evaluation.eval_time=0.005",
                f"++workload.train.epochs={epochs}",
                f"++workload.checkpoint.epochs_between_checkpoints={epoch_per_ckp}",
                f"++workload.model.model_size={model_size}",
                f"++workload.model.optimization_groups={optimizers}",
                f"++workload.model.num_layers={num_layers}",
                f"++workload.model.parallelism.zero_stage={zero_stage}",
                f"++workload.model.layer_parameters={layer_params}",
                f"++workload.model.parallelism.tensor={comm.size}",
                "++workload.dataset.format=indexed_binary",
                "++workload.dataset.num_files_train=4",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_samples_per_file=4",
                "++workload.dataset.record_length_bytes=256",
                "++workload.reader.read_threads=1",
                "++workload.reader.multiprocessing_context=fork",
            ],
        )
        run_benchmark(cfg, verify=True)

    if comm.rank == 0:
        ptmsc_keys = [k for k in store if k.endswith(".ptmsc")]
        nranks = comm.size
        num_model_files = 1
        num_optimizer_files = 1
        num_layer_files = 1
        files_per_checkpoint = (num_model_files + num_optimizer_files + num_layer_files) * nranks
        num_check_files = epochs / epoch_per_ckp * files_per_checkpoint
        assert len(ptmsc_keys) == num_check_files, f"got {len(ptmsc_keys)} {ptmsc_keys}"

    finalize()
    _clean_msc_prefix(store, storage_root)
