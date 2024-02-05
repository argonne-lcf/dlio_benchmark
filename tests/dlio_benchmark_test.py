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
import shutil
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
config_dir=os.path.dirname(dlio_benchmark.__file__)+"/configs/"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_benchmark_test.log", mode="a", encoding='utf-8'),
        logging.StreamHandler()
    ], format='[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
    # logging's max timestamp resolution is msecs, we will pass in usecs in the message
)

from dlio_benchmark.main import DLIOBenchmark
import glob

def init():
    DLIOMPI.get_instance().initialize()

def finalize():
    DLIOMPI.get_instance().finalize()

def clean(storage_root="./") -> None:
    comm.Barrier()
    if (comm.rank == 0):
        shutil.rmtree(os.path.join(storage_root, "checkpoints"), ignore_errors=True)
        shutil.rmtree(os.path.join(storage_root, "data/"), ignore_errors=True)
        shutil.rmtree(os.path.join(storage_root, "output"), ignore_errors=True)
    comm.Barrier()


def run_benchmark(cfg, storage_root="./", verify=True):

    comm.Barrier()
    if (comm.rank == 0):
        shutil.rmtree(os.path.join(storage_root, "output"), ignore_errors=True)
    comm.Barrier()
    t0 = time.time()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg['workload'])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    t1 = time.time()
    if (comm.rank==0):
        logging.info("Time for the benchmark: %.10f" %(t1-t0)) 
        if (verify):
            assert(len(glob.glob(benchmark.output_folder+"./*_output.json"))==benchmark.comm_size)
    return benchmark


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("fmt, framework", [("png", "tensorflow"), ("npz", "tensorflow"),
                                            ("jpeg", "tensorflow"), ("tfrecord", "tensorflow"),
                                            ("hdf5", "tensorflow"), ("indexed_binary", "tensorflow"), ("mmap_indexed_binary", "tensorflow")])
def test_gen_data(fmt, framework) -> None:
    init()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for generating {fmt} dataset")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=[f'++workload.framework={framework}',
                                                       f'++workload.reader.data_loader={framework}',
                                                       '++workload.workflow.train=False',
                                                       '++workload.workflow.generate_data=True',
                                                       f"++workload.dataset.format={fmt}", 
                                                       "++workload.dataset.num_files_train=8", 
                                                       "++workload.dataset.num_files_eval=8"])
        benchmark = run_benchmark(cfg, verify=False)
        if benchmark.args.num_subfolders_train <= 1:
            train = pathlib.Path(f"{cfg.workload.dataset.data_folder}/train")
            train_files = list(train.glob(f"*.{fmt}"))
            valid = pathlib.Path(f"{cfg.workload.dataset.data_folder}/valid")
            valid_files = list(valid.glob(f"*.{fmt}"))
            assert (len(train_files) == cfg.workload.dataset.num_files_train)
            assert (len(valid_files) == cfg.workload.dataset.num_files_eval)
        else:
            train = pathlib.Path(f"{cfg.workload.dataset.data_folder}/train")
            train_files = list(train.rglob(f"**/*.{fmt}"))
            valid = pathlib.Path(f"{cfg.workload.dataset.data_folder}/valid")
            valid_files = list(valid.rglob(f"**/*.{fmt}"))
            assert (len(train_files) == cfg.workload.dataset.num_files_train)
            assert (len(valid_files) == cfg.workload.dataset.num_files_eval)
        clean()
    finalize()

@pytest.mark.timeout(60, method="thread")
def test_subset() -> None:
    init()
    clean()
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO training test for subset")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=False', \
                    '++workload.workflow.generate_data=True'])
        benchmark=run_benchmark(cfg, verify=False)
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                        '++workload.workflow.generate_data=False', \
                            '++workload.dataset.num_files_train=8', \
                            '++workload.train.computation_time=0.01'])
        benchmark=run_benchmark(cfg, verify=True)
    clean()
    finalize()

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("fmt, framework", [("png", "tensorflow"), ("npz", "tensorflow"),
                                            ("jpeg", "tensorflow"), ("tfrecord", "tensorflow"),
                                            ("hdf5", "tensorflow"), ("indexed_binary", "tensorflow"),
                                            ("mmap_indexed_binary", "tensorflow")])
def test_storage_root_gen_data(fmt, framework) -> None:
    init()
    storage_root = "runs"

    clean(storage_root)
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for generating {fmt} dataset")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=[f'++workload.framework={framework}',
                                                       f'++workload.reader.data_loader={framework}',
                                                       '++workload.workflow.train=False',
                                                       '++workload.workflow.generate_data=True',
                                                       f"++workload.storage.storage_root={storage_root}",
                                                       f"++workload.dataset.format={fmt}", 
                                                       "++workload.dataset.num_files_train=16"])
        benchmark = run_benchmark(cfg, verify=False)
        if benchmark.args.num_subfolders_train <= 1:
            assert (
                    len(glob.glob(
                        os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*.{fmt}"))) ==
                    cfg.workload.dataset.num_files_train)
            assert (
                    len(glob.glob(
                        os.path.join(storage_root, cfg.workload.dataset.data_folder, f"valid/*.{fmt}"))) ==
                    cfg.workload.dataset.num_files_eval)
        else:
            logging.info(os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*/*.{fmt}"))
            assert (
                    len(glob.glob(
                        os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*/*.{fmt}"))) ==
                    cfg.workload.dataset.num_files_train)
            assert (
                    len(glob.glob(
                        os.path.join(storage_root, cfg.workload.dataset.data_folder, f"valid/*/*.{fmt}"))) ==
                    cfg.workload.dataset.num_files_eval)
        clean(storage_root)
    finalize()

@pytest.mark.timeout(60, method="thread")
def test_iostat_profiling() -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for iostat profiling")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=False',
                                                       '++workload.workflow.generate_data=True'])

        benchmark = run_benchmark(cfg, verify=False)
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
                                                       '++workload.workflow.generate_data=False',
                                                       'workload.train.computation_time=0.01',
                                                       'workload.evaluation.eval_time=0.005',
                                                       'workload.train.epochs=1',
                                                       'workload.workflow.profiling=True',
                                                       'workload.profiling.profiler=iostat'])
        benchmark = run_benchmark(cfg)
        assert (os.path.isfile(benchmark.output_folder + "/iostat.json"))
        if (comm.rank == 0):
            logging.info("generating output data")
            hydra = f"{benchmark.output_folder}/.hydra"
            os.makedirs(hydra, exist_ok=True)
            yl: str = OmegaConf.to_yaml(cfg)
            with open(f"{hydra}/config.yaml", "w") as f:
                OmegaConf.save(cfg, f)
            with open(f"{hydra}/overrides.yaml", "w") as f:
                f.write('[]')
            subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)
            cmd = f"dlio_postprocessor --output-folder={benchmark.output_folder}"
            cmd = cmd.split()
            subprocess.run(cmd, capture_output=True, timeout=10)
        clean()
    finalize()

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("framework, model_size, optimizers, num_layers, layer_params, type", [("tensorflow", 1024, [1024, 128], 2, [16], "all_ranks"),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], "all_ranks"),
                                                                                         ("tensorflow", 1024, [1024, 128], 2, [16], "rank_zero"),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], "rank_zero"),
                                                                                         ("tensorflow", 1024, [128], 1, [], "all_ranks"),
                                                                                         ("pytorch", 1024, [128], 1, [], "all_ranks")])
def test_checkpoint_epoch(framework, model_size, optimizers, num_layers, layer_params, type) -> None:
    init()
    clean()
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for checkpointing at the end of epochs")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config',
                      overrides=[f'++workload.framework={framework}',
                                 f'++workload.reader.data_loader={framework}',
                                 '++workload.workflow.train=True',
                                 '++workload.workflow.generate_data=True',
                                 '++workload.train.computation_time=0.01',
                                 '++workload.evaluation.eval_time=0.005',
                                 '++workload.train.epochs=8', '++workload.workflow.checkpoint=True',
                                 '++workload.checkpoint.epochs_between_checkpoints=2',
                                 f'++workload.checkpoint.type={type}',
                                 f'++workload.checkpoint.model_size={model_size}',
                                 f'++workload.checkpoint.optimization_groups={optimizers}',
                                 f'++workload.checkpoint.num_layers={num_layers}',
                                 f'++workload.checkpoint.layer_parameters={layer_params}'])
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree("./checkpoints", ignore_errors=True)
            os.makedirs("./checkpoints", exist_ok=True)
        comm.Barrier()
        benchmark = run_benchmark(cfg)
        output = pathlib.Path("./checkpoints")
        load_bin = list(output.glob("*"))
        n = 0
        if len(layer_params) > 0:
            n = num_layers
        nranks = 1
        if type == "all_ranks":
            nranks = comm.size
        if framework == "tensorflow":
            num_check_files = 8 / 2 * (2 + 2 + 2*n) * nranks + 1
            assert (len(load_bin) == num_check_files), f"files produced are {len(load_bin)} {num_check_files} {load_bin} "
        if framework == "pytorch":
            num_check_files = 8 / 2 * (1 + 1 + n) * nranks
            assert (len(load_bin) == num_check_files), f"files produced are {len(load_bin)} {num_check_files} {load_bin}"
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree("./checkpoints", ignore_errors=True)
        comm.Barrier()
        clean()
    finalize()

@pytest.mark.timeout(60, method="thread")
def test_checkpoint_step() -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for checkpointing at the end of steps")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config',
                      overrides=['++workload.workflow.train=True', \
                                 '++workload.workflow.generate_data=True', \
                                 '++workload.train.computation_time=0.01', \
                                 '++workload.evaluation.eval_time=0.005', \
                                 '++workload.train.epochs=8', '++workload.workflow.checkpoint=True', \
                                 '++workload.checkpoint.steps_between_checkpoints=2'])
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree("./checkpoints", ignore_errors=True)
            os.makedirs("./checkpoints", exist_ok=True)
        comm.Barrier()
        benchmark = run_benchmark(cfg)
        dataset = cfg['workload']['dataset']
        nstep = dataset.num_files_train * dataset.num_samples_per_file // cfg['workload'][
            'reader'].batch_size // benchmark.comm_size
        ncheckpoints = nstep // 2 * 8 * 2
        output = pathlib.Path("./checkpoints")
        load_bin = list(output.glob("*"))
        assert (len(load_bin) == ncheckpoints)
        clean()
    finalize()

@pytest.mark.timeout(60, method="thread")
def test_eval() -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for evaluation")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config',
                      overrides=['++workload.workflow.train=True', \
                                 '++workload.workflow.generate_data=True', \
                                 'workload.train.computation_time=0.01', \
                                 'workload.evaluation.eval_time=0.005', \
                                 '++workload.train.epochs=4', '++workload.workflow.evaluation=True'])
        benchmark = run_benchmark(cfg)
        clean()
    finalize()

@pytest.mark.timeout(60, method="thread")

@pytest.mark.parametrize("framework, nt", [("tensorflow", 0), ("tensorflow", 1),("tensorflow", 2),
                                           ("pytorch", 0), ("pytorch", 1), ("pytorch", 2)])
def test_multi_threads(framework, nt) -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for generating multithreading read_threads={nt} {framework} framework")
        logging.info("=" * 80)
        # with subTest(f"Testing full benchmark for format: {framework}-NT{nt}", nt=nt, framework=framework):
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
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
    clean()
    finalize()

@pytest.mark.timeout(60, method="thread")

@pytest.mark.parametrize("nt, context", [(0, None), (1, "fork"), (2, "spawn"), (2, "forkserver")])
def test_pytorch_multiprocessing_context(nt, context) -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for pytorch multiprocessing_context={context} read_threads={nt}")
        logging.info("=" * 80)
        # with subTest(f"Testing full benchmark for format: {framework}-NT{nt}", nt=nt, framework=pytorch):
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
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
    clean()
    finalize()

@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("fmt, framework, dataloader", [("png", "tensorflow","tensorflow"), ("npz", "tensorflow","tensorflow"),
                                            ("jpeg", "tensorflow","tensorflow"), ("tfrecord", "tensorflow","tensorflow"),
                                            ("hdf5", "tensorflow","tensorflow"), ("csv", "tensorflow","tensorflow"),
                                            ("indexed_binary", "tensorflow","tensorflow"), ("mmap_indexed_binary", "tensorflow","tensorflow"),
                                            ("png", "pytorch", "pytorch"), ("npz", "pytorch", "pytorch"),
                                            ("jpeg", "pytorch", "pytorch"), ("hdf5", "pytorch", "pytorch"),
                                            ("csv", "pytorch", "pytorch"), ("indexed_binary", "pytorch", "pytorch"),
                                            ("mmap_indexed_binary", "pytorch", "pytorch"),
                                            ("png", "tensorflow", "dali"), ("npz", "tensorflow", "dali"),
                                            ("jpeg", "tensorflow", "dali"), ("hdf5", "tensorflow", "dali"),
                                            ("csv", "tensorflow", "dali"), ("indexed_binary", "tensorflow", "dali"),
                                            ("mmap_indexed_binary", "tensorflow", "dali"),
                                            ("png", "pytorch", "dali"), ("npz", "pytorch", "dali"),
                                            ("jpeg", "pytorch", "dali"), ("hdf5", "pytorch", "dali"),
                                            ("csv", "pytorch", "dali"), ("indexed_binary", "pytorch", "dali"),
                                            ("mmap_indexed_binary", "pytorch", "dali"),
                                            ])
def test_train(fmt, framework, dataloader) -> None:
    init()
    clean()
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO training test: Generating data for {fmt} format")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
                                                       '++workload.workflow.generate_data=True',
                                                       f"++workload.framework={framework}", \
                                                       f"++workload.reader.data_loader={dataloader}", \
                                                       f"++workload.dataset.format={fmt}",
                                                       'workload.train.computation_time=0.01', \
                                                       'workload.evaluation.eval_time=0.005', \
                                                       '++workload.train.epochs=1', \
                                                       '++workload.dataset.num_files_train=16', \
                                                       '++workload.reader.read_threads=1'])
        benchmark = run_benchmark(cfg)
    #clean()
    finalize()


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("fmt, framework", [("png", "tensorflow"), ("npz", "tensorflow"),
                                            ("jpeg", "tensorflow"), ("tfrecord", "tensorflow"),
                                            ("hdf5", "tensorflow"), ("csv", "tensorflow"),
                                            ("indexed_binary", "tensorflow"), ("mmap_indexed_binary", "tensorflow"),
                                            ("png", "pytorch"), ("npz", "pytorch"),
                                            ("jpeg", "pytorch"), ("hdf5", "pytorch"),
                                            ("csv", "pytorch"), ("indexed_binary", "pytorch"),
                                            ("mmap_indexed_binary", "pytorch"),
                                            ])
def test_custom_storage_root_train(fmt, framework) -> None:
    init()
    storage_root = "root_dir"
    clean(storage_root)
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO training test for {fmt} format in {framework} framework")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                       '++workload.workflow.generate_data=True', \
                                                       f"++workload.framework={framework}", \
                                                       f"++workload.reader.data_loader={framework}", \
                                                       f"++workload.dataset.format={fmt}",
                                                       f"++workload.storage.storage_root={storage_root}", \
                                                       'workload.train.computation_time=0.01', \
                                                       'workload.evaluation.eval_time=0.005', \
                                                       '++workload.train.epochs=1', \
                                                       '++workload.dataset.num_files_train=16', \
                                                       '++workload.reader.read_threads=1'])
        benchmark = run_benchmark(cfg)
    clean(storage_root)
    finalize()

if __name__ == '__main__':
    unittest.main()
