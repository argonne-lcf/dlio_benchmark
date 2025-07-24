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
import os
import glob
from datetime import datetime
from collections import Counter

from mpi4py import MPI

from hydra import initialize_config_dir, compose

from dlio_benchmark.main import DLIOBenchmark
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
import dlio_benchmark

comm = MPI.COMM_WORLD

config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

def run_benchmark(cfg):
    comm.Barrier()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg["workload"])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    return benchmark

def delete_folder(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


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

def check_ai_events(path):
    counter = Counter(root=0, compute=0, item=0, preprocess=0, fetch_iter=0, train=0, eval=0, epoch=0, ckpt_capture=0, ckpt_restart=0)
    with open(path, mode="r") as f:
        for line in f:
            if "[" in line or "]" in line:
                continue
            if '"cat":"ai_root"' in line and '"name":"ai_root"' in line:
                counter["root"] += 1
            if '"cat":"compute"' in line and '"name":"compute"' in line:
                counter["compute"] += 1
            if '"cat":"data"' in line and '"name":"item"' in line:
                counter["item"] += 1
            if '"cat":"data"' in line and '"name":"preprocess"' in line:
                counter["preprocess"] += 1
            if '"cat":"dataloader"' in line and '"name":"fetch.iter"' in line:
                counter["fetch_iter"] += 1
            if '"cat":"checkpoint"' in line and '"name":"capture"' in line:
                counter["ckpt_capture"] += 1
            if '"cat":"checkpoint"' in line and '"name":"restart"' in line:
                counter["ckpt_restart"] += 1
            if '"cat":"pipeline"' in line and '"name":"train"' in line:
                counter["train"] += 1
            if '"cat":"pipeline"' in line and '"name":"evaluate"' in line:
                counter["eval"] += 1
            if '"cat":"pipeline"' in line and '"name":"epoch.block"' in line:
                counter["epoch"] += 1
    return counter

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework, num_data, batch_size", [
    (framework, num_data, batch_size)
    for framework in ["pytorch", "tensorflow"]
    for num_data in [9, 10]  # even and odd
    for batch_size in [2, 3]  # even and odd
])
def test_ai_logging_train(setup_test_env, framework, num_data, batch_size):
    storage_root = setup_test_env
    num_epochs = 2
    num_data_pp = num_data
    total_data = num_data_pp * comm.size
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=True",
                "++workload.workflow.evaluation=False",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={total_data}",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_subfolders_train=0",
                "++workload.dataset.num_subfolders_eval=0",
                f"++workload.train.epochs={num_epochs}",
                f"++workload.reader.batch_size={batch_size}"
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))

    assert len(paths) > 0, "No pfw files found"

    count = check_ai_events(path=paths[comm.rank % len(paths)])

    if comm.rank == 0:
        print("AI events count:", count)

    # check single file from single rank only
    assert count["root"]       == 1
    assert count["epoch"]      == num_epochs
    assert count["train"]      == num_epochs
    assert count["eval"]       == 0
    # @ray: we are using // because in DLIO we always drop last when
    # number of data is not evenly divisible by the batch size
    assert count["fetch_iter"] == num_epochs * (num_data_pp // batch_size)
    assert count["compute"]    == num_epochs * (num_data_pp // batch_size)

    # @ray: this is the tricky part, we run data fetching on background
    # (e.g. using parallel workers). using exact number will be difficult
    # we use relax comparison and approximation (+/- 2) instead.
    assert count["item"]       >= num_epochs * (num_data_pp // batch_size)
    assert count["preprocess"] >= num_epochs * (num_data_pp // batch_size)
    
    assert count["ckpt_capture"] == 0
    assert count["ckpt_restart"] == 0

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework, step, read_threads", [
    (framework, step, read_threads)
    for framework in ["pytorch", "tensorflow"]
    for step in [2, 3]  # even and odd
    for read_threads in [2, 3]  # even and odd
])
def test_ai_logging_train_with_step(setup_test_env, framework, step, read_threads):
    storage_root = setup_test_env
    num_epochs = 2
    batch_size = 2
    num_data_pp = 8
    total_data = num_data_pp * comm.size
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=True",
                "++workload.workflow.evaluation=False",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={total_data}",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_subfolders_train=0",
                "++workload.dataset.num_subfolders_eval=0",
                f"++workload.reader.batch_size={batch_size}",
                f"++workload.train.epochs={num_epochs}",
                f"++workload.train.total_training_steps={step}",
                f"++workload.reader.read_threads={read_threads}",
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))

    assert len(paths) > 0, "No pfw files found"

    count = check_ai_events(path=paths[comm.rank % len(paths)])

    if comm.rank == 0:
        print("AI events count:", count)

    # check single file from single rank only
    assert count["root"]       == 1
    assert count["epoch"]      == num_epochs
    assert count["train"]      == num_epochs
    assert count["eval"]       == 0
    assert count["fetch_iter"] == num_epochs * step
    assert count["compute"]    == num_epochs * step

    # @ray: we are using relax comparison and approximation (+/- 2)
    assert count["item"]       >= num_epochs * step
    assert count["preprocess"] >= num_epochs * step

    assert count["ckpt_capture"] == 0
    assert count["ckpt_restart"] == 0


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_ai_logging_with_eval(setup_test_env, framework):
    storage_root = setup_test_env
    num_epochs = 2
    batch_size = 1
    num_data_pp = 8
    total_data = num_data_pp * comm.size
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=True",
                "++workload.workflow.evaluation=True",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={total_data}",
                f"++workload.dataset.num_files_eval={total_data}",
                "++workload.dataset.num_subfolders_train=0",
                "++workload.dataset.num_subfolders_eval=0",
                f"++workload.reader.batch_size={batch_size}",
                f"++workload.train.epochs={num_epochs}"
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))
    assert len(paths) > 0, "No pfw files found"

    count = check_ai_events(path=paths[comm.rank % len(paths)])

    if comm.rank == 0:
        print("AI events count:", count)

    # check single file from single rank only
    assert count["root"]         == 1
    assert count["epoch"]        == num_epochs
    assert count["train"]        == num_epochs
    assert count["eval"]         == num_epochs
    assert count["fetch_iter"]   == 2 * num_epochs * (num_data_pp // batch_size)
    assert count["compute"]      == 2 * num_epochs * (num_data_pp // batch_size)

    # @ray: we are using relax comparison and approximation (+/- 2)
    assert count["item"]         == 2 * num_epochs * num_data_pp
    assert count["preprocess"]   == 2 * num_epochs * num_data_pp

    assert count["ckpt_capture"] == 0
    assert count["ckpt_restart"] == 0

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework, fmt", [
    (framework, fmt) 
    for framework in ["pytorch", "tensorflow"]
    for fmt in ["hdf5", "npy", "npz", "tfrecord", "csv", "jpeg", "png", "indexed_binary", "mmap_indexed_binary", "synthetic"]
    if not (fmt == "tfrecord" and framework == "pytorch")  # Exclude tfrecord + pytorch
])
def test_ai_logging_with_reader(setup_test_env, framework, fmt):
    storage_root = setup_test_env
    num_epochs = 2
    batch_size = 1
    num_data_pp = 8
    total_data = num_data_pp * comm.size
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.train=True",
                "++workload.workflow.evaluation=True",
                "++workload.workflow.generate_data=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={total_data}",
                f"++workload.dataset.num_files_eval={total_data}",
                "++workload.dataset.num_subfolders_train=0",
                "++workload.dataset.num_subfolders_eval=0",
                f"++workload.reader.batch_size={batch_size}",
                f"++workload.train.epochs={num_epochs}",
                f"++workload.dataset.format={fmt}",
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))

    assert len(paths) > 0, "No pfw files found"

    count = check_ai_events(path=paths[comm.rank % len(paths)])

    if comm.rank == 0:
        print("AI events count:", count)

    assert count["root"]       == 1
    assert count["epoch"]      == num_epochs
    assert count["train"]      == num_epochs
    assert count["eval"]       == num_epochs
    assert count["fetch_iter"] == 2 * num_epochs * (num_data_pp // batch_size)
    assert count["compute"]    == 2 * num_epochs * (num_data_pp // batch_size)
    if fmt == "tfrecord":
        # @ray: tfrecord reader does not have notion of data item since our function
        # will be fused into execution graph, making it impossible to count the events
        # by just using decorator in python
        assert count["item"] == 0
        assert count["preprocess"] == 0
    else:
        # @ray: we are using relax comparison and approximation (+/- 2)
        assert count["item"]       == 2 * num_epochs * num_data_pp
        if fmt == "synthetic":
            # @ray: synthetic reader has no preprocess
            assert count["preprocess"] == 0
        else:
            assert count["preprocess"] == 2 * num_epochs * num_data_pp

    assert count["ckpt_capture"] == 0
    assert count["ckpt_restart"] == 0

# @ray: future note: it seems DLIO hasn't implemented the all_ranks checkpointing yet
# this test suite is only for checkpointing on rank_zero only
# @todo: add test-cases to test all_ranks by adding ++workload.checkpoint.type=<VALUE>
@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework, epoch_per_ckpt, steps_per_ckpt", [
    (framework, epoch_per_ckpt, steps_per_ckpt)
    for framework in ["pytorch", "tensorflow"]
    for epoch_per_ckpt in [1, 2]
    for steps_per_ckpt in ["na", 1, 2]
])
def test_ai_logging_train_with_checkpoint(setup_test_env, framework, epoch_per_ckpt, steps_per_ckpt):
    storage_root = setup_test_env
    num_epochs = 2
    batch_size = 1
    num_data_pp = 4
    total_data = num_data_pp * comm.size
    if steps_per_ckpt == "na":
        steps_per_ckpt = -1
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.generate_data=True",
                "++workload.workflow.train=True",
                "++workload.workflow.evaluation=False",
                "++workload.workflow.checkpoint=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                f"++workload.dataset.num_files_train={total_data}",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_subfolders_train=0",
                "++workload.dataset.num_subfolders_eval=0",
                f"++workload.train.epochs={num_epochs}",
                f"++workload.reader.batch_size={batch_size}",
                f"++workload.checkpoint.epochs_between_checkpoints={epoch_per_ckpt}",
                f"++workload.checkpoint.steps_between_checkpoints={steps_per_ckpt}",
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))

    assert len(paths) > 0, "No pfw files found"

    # find trace-{RANK} in paths
    chosen_path = [p for p in paths if f"trace-{comm.rank}" in p]
    assert len(chosen_path) == 1
    path = chosen_path[0]

    count = check_ai_events(path=path)

    print(f"[RANK-{comm.rank}] AI events count: {count}")

    assert count["root"]       == 1
    assert count["epoch"]      == num_epochs
    assert count["train"]      == num_epochs
    assert count["eval"]       == 0
    assert count["fetch_iter"] == num_epochs * (num_data_pp // batch_size)
    assert count["compute"]    == num_epochs * (num_data_pp // batch_size)

    # @ray: we are using relax comparison and approximation (+/- 2)
    assert count["item"]       >= num_epochs * (num_data_pp // batch_size)
    assert count["preprocess"] >= num_epochs * (num_data_pp // batch_size)

    # @ray: this assertion below is only for rank 0
    # @todo: when DLIO supports all_ranks checkpointing, adjust this
    ckpt_capture = 0
    if comm.rank == 0:
        ckpt_capture = count["ckpt_capture"]
    else:
        ckpt_capture = None

    ckpt_capture = comm.bcast(ckpt_capture, root=0)
    assert ckpt_capture is not None

    # @ray: in DLIO step has more precedence compared to epoch
    if steps_per_ckpt != -1:
        expected_checkpoints = num_epochs * (num_data_pp // batch_size) // steps_per_ckpt
    else:
        expected_checkpoints = num_epochs // epoch_per_ckpt

    assert ckpt_capture == expected_checkpoints
    assert count["ckpt_restart"] == 0

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework, num_checkpoint_write, num_checkpoint_read", [
    (framework, num_checkpoint_write, num_checkpoint_read)
    for framework in ["pytorch", "tensorflow"]
    for num_checkpoint_write in [3, 4]
    for num_checkpoint_read in [1, 2, 3]
])
def test_ai_logging_checkpoint_only(setup_test_env, framework, num_checkpoint_write, num_checkpoint_read):
    storage_root = setup_test_env
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.framework={framework}",
                f"++workload.reader.data_loader={framework}",
                "++workload.workflow.generate_data=False",
                "++workload.workflow.train=False",
                "++workload.workflow.evaluation=False",
                "++workload.workflow.checkpoint=True",
                f"++workload.output.folder={storage_root}",
                f"++workload.dataset.data_folder={storage_root}/data",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_subfolders_train=0",
                "++workload.dataset.num_subfolders_eval=0",
                f"++workload.checkpoint.checkpoint_folder={storage_root}/checkpoint",
                f"++workload.checkpoint.num_checkpoints_write={num_checkpoint_write}",
                f"++workload.checkpoint.num_checkpoints_read={num_checkpoint_read}",
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))

    assert len(paths) > 0, "No pfw files found"

    # find trace-{RANK} in paths
    chosen_path = [p for p in paths if f"trace-{comm.rank}" in p]
    assert len(chosen_path) == 1
    path = chosen_path[0]

    count = check_ai_events(path=path)

    print(f"[RANK-{comm.rank}] AI events count: {count}")

    assert count["root"]       == 1
    assert count["epoch"]      == 0
    assert count["train"]      == 0
    assert count["eval"]       == 0
    assert count["fetch_iter"] == 0
    assert count["item"]       == 0
    assert count["preprocess"] == 0

    # @ray: this assertion below is only for rank 0
    # @todo: when DLIO supports all_ranks checkpointing, adjust this
    ckpt_capture = 0
    ckpt_restart = 0
    if comm.rank == 0:
        ckpt_capture = count["ckpt_capture"]
        ckpt_restart = count["ckpt_restart"]
    else:
        ckpt_capture = None
        ckpt_restart = None

    ckpt_capture = comm.bcast(ckpt_capture, root=0)
    ckpt_restart = comm.bcast(ckpt_restart, root=0)
    assert ckpt_capture is not None
    assert ckpt_restart is not None
    assert ckpt_capture == num_checkpoint_write
    assert ckpt_restart == num_checkpoint_read
    assert count["compute"] == num_checkpoint_write + num_checkpoint_read