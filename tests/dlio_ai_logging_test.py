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
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        storage_root = os.path.join("outputs", f"{now}-{str(uuid.uuid4())}")
    else:
        storage_root = None
    storage_root = comm.bcast(storage_root, root=0)

    if comm.rank == 0:
        if os.path.exists(storage_root):
            shutil.rmtree(storage_root)
        os.makedirs(storage_root, exist_ok=True)

    comm.Barrier()
    yield storage_root
    # comm.Barrier()
    # if comm.rank == 0:
    #     shutil.rmtree(storage_root, ignore_errors=True)
    comm.Barrier()
    DLIOMPI.get_instance().finalize()

def check_ai_events(path):
    counter = Counter(root=0, compute=0, data_item=0, fetch_iter=0, train=0, eval=0, epoch=0)
    with open(path, mode="r") as f:
        for line in f:
            if "[" in line or "]" in line:
                continue
            if '"cat":"ai_root"' in line and '"name":"ai_root"' in line:
                counter["root"] += 1
            if '"cat":"compute"' in line and '"name":"compute"' in line:
                counter["compute"] += 1
            if '"cat":"data"' in line and '"name":"item"' in line:
                counter["data_item"] += 1
            if '"cat":"dataloader"' in line and '"name":"fetch.iter"' in line:
                counter["fetch_iter"] += 1
            if '"cat":"pipeline"' in line and '"name":"train"' in line:
                counter["train"] += 1
            if '"cat":"pipeline"' in line and '"name":"evaluate"' in line:
                counter["eval"] += 1
            if '"cat":"pipeline"' in line and '"name":"epoch.block"' in line:
                counter["epoch"] += 1
    return counter

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_ai_logging_train(setup_test_env, framework):
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
                f"++workload.train.epochs={num_epochs}",
                f"++workload.reader.batch_size={batch_size}"
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))
    if len(paths) == 0:
        pytest.fail("No pfw files found")

    count = check_ai_events(path=paths[0])

    if comm.rank == 0:
        print("AI events count:", count)

    # check single file from single rank only
    assert count["root"]       == 1
    assert count["epoch"]      == num_epochs
    assert count["train"]      == num_epochs
    assert count["eval"]       == 0
    assert count["fetch_iter"] == num_epochs * (num_data_pp // batch_size)
    assert count["compute"]    == num_epochs * (num_data_pp // batch_size)

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework, step", [("pytorch", 3), ("tensorflow", 3)])
def test_ai_logging_train_with_step(setup_test_env, framework, step):
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
            ],
        )
        run_benchmark(cfg)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))
    if len(paths) == 0:
        pytest.fail("No pfw files found")

    count = check_ai_events(path=paths[0])

    if comm.rank == 0:
        print("AI events count:", count)

    # check single file from single rank only
    assert count["root"]       == 1
    assert count["epoch"]      == num_epochs
    assert count["train"]      == num_epochs
    assert count["eval"]       == 0
    assert count["fetch_iter"] == num_epochs * (step // batch_size)
    assert count["compute"]    == num_epochs * (step // batch_size)


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
    if len(paths) == 0:
        pytest.fail("No pfw files found")

    count = check_ai_events(path=paths[0])

    if comm.rank == 0:
        print("AI events count:", count)

    # check single file from single rank only
    assert count["root"]       == 1
    assert count["epoch"]      == num_epochs
    assert count["train"]      == num_epochs
    assert count["eval"]       == num_epochs
    assert count["fetch_iter"] == 2 * num_epochs * (num_data_pp // batch_size)
    assert count["compute"]    == 2 * num_epochs * (num_data_pp // batch_size)