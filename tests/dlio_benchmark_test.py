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
from collections import namedtuple
from src.utils.utility import timeit
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import hydra
import unittest
import yaml
import shutil
from mpi4py import MPI
comm = MPI.COMM_WORLD
import pytest
import time
import subprocess
import logging
import os
from src.utils.utility import ConfigArguments

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_benchmark_test.log", mode = "a", encoding='utf-8'),
        logging.StreamHandler()
    ],format='[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'  # logging's max timestamp resolution is msecs, we will pass in usecs in the message
)
 
from src.dlio_benchmark import DLIOBenchmark
import glob
class TestDLIOBenchmark(unittest.TestCase):
    def clean(self, storage_root= "./") -> None:
        comm.Barrier()
        if (comm.rank==0):
            shutil.rmtree(os.path.join(storage_root, "checkpoints"), ignore_errors=True)
            shutil.rmtree(os.path.join(storage_root, "data/"), ignore_errors=True)
            shutil.rmtree(os.path.join(storage_root, "output"), ignore_errors=True)
        comm.Barrier()
      
    def run_benchmark(self, cfg, storage_root="./", verify=True):
        comm.Barrier()
        if (comm.rank==0):
            shutil.rmtree(os.path.join(storage_root, "output"), ignore_errors=True)
        comm.Barrier()     
        t0=time.time()
        args = ConfigArguments.get_instance()
        args.reset()
        benchmark = DLIOBenchmark(cfg['workload'])
        benchmark.initialize()
        benchmark.run()
        benchmark.finalize()
        t1=time.time()
        if (comm.rank==0):
            logging.info("Time for the benchmark: %.10f" %(t1-t0)) 
        if (verify):
            self.assertEqual(len(glob.glob(benchmark.output_folder+"./*_load_and_proc_times.json")), benchmark.comm_size)
        return benchmark
    @pytest.mark.timeout(60, method="thread")        
    def test_gen_data(self) -> None:
        for fmt in "tfrecord", "jpeg", "png", "hdf5", "npz":
            with self.subTest(f"Testing data generator for format: {fmt}", fmt=fmt):
                if (comm.rank==0):
                    logging.info("")
                    logging.info("="*80)
                    logging.info(f" DLIO test for generating {fmt} dataset")
                    logging.info("="*80)
                with initialize(version_base=None, config_path="../configs"):
                    cfg = compose(config_name='config', overrides=['++workload.workflow.train=False', \
                                                                   '++workload.workflow.generate_data=True',
                                                                   f"++workload.dataset.format={fmt}"])
                    benchmark=self.run_benchmark(cfg, verify=False)
                    if benchmark.args.num_subfolders_train<=1:
                        self.assertEqual(len(glob.glob(os.path.join(cfg.workload.dataset.data_folder, f"train/*.{fmt}"))), cfg.workload.dataset.num_files_train)
                        self.assertEqual(len(glob.glob(os.path.join(cfg.workload.dataset.data_folder,  f"valid/*.{fmt}"))), cfg.workload.dataset.num_files_eval)
                    else:
                        self.assertEqual(len(glob.glob(os.path.join(cfg.workload.dataset.data_folder, f"train/*/*.{fmt}"))), cfg.workload.dataset.num_files_train)
                        self.assertEqual(len(glob.glob(os.path.join(cfg.workload.dataset.data_folder, f"valid/*/*.{fmt}"))), cfg.workload.dataset.num_files_eval)
                    self.clean()

    @pytest.mark.timeout(60, method="thread")
    def test_storage_root_gen_data(self) -> None:
        storage_root="runs"
        for fmt in "tfrecord", "jpeg", "png", "hdf5", "npz":
            with self.subTest(f"Testing data generator for format: {fmt}", fmt=fmt):
                self.clean(storage_root)
                if (comm.rank==0):
                    logging.info("")
                    logging.info("="*80)
                    logging.info(f" DLIO test for generating {fmt} dataset")
                    logging.info("="*80)
                with initialize(version_base=None, config_path="../configs"):
                    cfg = compose(config_name='config', overrides=['++workload.workflow.train=False', \
                                                                   '++workload.workflow.generate_data=True',
                                                                   f"++workload.storage.storage_root={storage_root}",
                                                                   f"++workload.dataset.format={fmt}"])
                    benchmark=self.run_benchmark(cfg, verify=False)
                    if benchmark.args.num_subfolders_train<=1:
                        self.assertEqual(len(glob.glob(os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*.{fmt}"))), cfg.workload.dataset.num_files_train)
                        self.assertEqual(len(glob.glob(os.path.join(storage_root, cfg.workload.dataset.data_folder,  f"valid/*.{fmt}"))), cfg.workload.dataset.num_files_eval)
                    else:
                        logging.info(os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*/*.{fmt}"))
                        self.assertEqual(len(glob.glob(os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*/*.{fmt}"))), cfg.workload.dataset.num_files_train)
                        self.assertEqual(len(glob.glob(os.path.join(storage_root, cfg.workload.dataset.data_folder, f"valid/*/*.{fmt}"))), cfg.workload.dataset.num_files_eval)
                    self.clean(storage_root)


    @pytest.mark.timeout(60, method="thread")
    def test_iostat_profiling(self) -> None:
        self.clean()
        if (comm.rank==0):
            logging.info("")
            logging.info("="*80)
            logging.info(f" DLIO test for iostat profiling")
            logging.info("="*80)
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config', overrides=['++workload.workflow.train=False',
                                                           '++workload.workflow.generate_data=True'])

            benchmark=self.run_benchmark(cfg, verify=False)
            cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=False',
                                                           'workload.train.computation_time=0.01',
                                                           'workload.evaluation.eval_time=0.005',
                                                           'workload.train.epochs=1',
                                                           'workload.workflow.profiling=True',
                                                           'workload.profiling.profiler=iostat'])
            benchmark=self.run_benchmark(cfg)
            assert(os.path.isfile(benchmark.output_folder+"/iostat.json"))
            if (comm.rank==0):
                logging.info("generating output data")
                os.makedirs(benchmark.output_folder+"/.hydra/", exist_ok=True)
                yl: str = OmegaConf.to_yaml(cfg)
                with open(benchmark.output_folder+"./.hydra/config.yaml", "w") as f:
                    OmegaConf.save(cfg, f)
                with open(benchmark.output_folder+"./.hydra/overrides.yaml", "w") as f:
                    f.write('[]')
                subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)
                cmd=f"python src/dlio_postprocessor.py --output-folder={benchmark.output_folder}"
                cmd=cmd.split()
                subprocess.run(cmd, capture_output=True, timeout=4)
            self.clean()

    @pytest.mark.timeout(60, method="thread")
    def test_checkpoint_epoch(self) -> None:
        self.clean()
        if (comm.rank==0):
                logging.info("")
                logging.info("="*80)
                logging.info(f" DLIO test for checkpointing at the end of epochs")
                logging.info("="*80)
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.workflow.train=True',\
                                     '++workload.workflow.generate_data=True', \
                                     '++workload.train.computation_time=0.01', \
                                     '++workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=8', '++workload.workflow.checkpoint=True', \
                                     '++workload.checkpoint.epochs_between_checkpoints=2'])
            comm.Barrier()
            if comm.rank==0:
                shutil.rmtree("./checkpoints", ignore_errors=True)
            comm.Barrier()
            benchmark=self.run_benchmark(cfg)            
            self.assertEqual(len(glob.glob("./checkpoints/*.bin")), 4)
            self.clean()

    @pytest.mark.timeout(60, method="thread")
    def test_checkpoint_step(self) -> None:
        self.clean()        
        if (comm.rank==0):
            logging.info("")
            logging.info("="*80)
            logging.info(f" DLIO test for checkpointing at the end of steps")
            logging.info("="*80)
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.workflow.train=True',\
                                     '++workload.workflow.generate_data=True', \
                                     '++workload.train.computation_time=0.01', \
                                     '++workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=8', '++workload.workflow.checkpoint=True', \
                                     '++workload.checkpoint.steps_between_checkpoints=2'])
            comm.Barrier()
            if comm.rank==0:
                shutil.rmtree("./checkpoints", ignore_errors=True)
            comm.Barrier()
            benchmark=self.run_benchmark(cfg)                        
            dataset = cfg['workload']['dataset']
            nstep = dataset.num_files_train * dataset.num_samples_per_file // cfg['workload']['reader'].batch_size//benchmark.comm_size
            ncheckpoints=nstep//2*8
            self.assertEqual(len(glob.glob("./checkpoints/*.bin")), ncheckpoints)
            self.clean()

    @pytest.mark.timeout(60, method="thread")
    def test_eval(self) -> None:
        self.clean()
        if (comm.rank==0):
            logging.info("")
            logging.info("="*80)
            logging.info(f" DLIO test for evaluation")
            logging.info("="*80)        
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.workflow.train=True',\
                                     '++workload.workflow.generate_data=True', \
                                     'workload.train.computation_time=0.01', \
                                     'workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=4', '++workload.workflow.evaluation=True'])
            benchmark=self.run_benchmark(cfg)      
            self.clean()

    @pytest.mark.timeout(60, method="thread")
    def test_multi_threads(self) -> None:
        self.clean()
        for framework in "tensorflow", "pytorch":
            for nt in 0, 1, 2:
                if (comm.rank==0):
                    logging.info("")
                    logging.info("="*80)
                    logging.info(f" DLIO test for generating multithreading read_threads={nt} {framework} framework")
                    logging.info("="*80)
                with self.subTest(f"Testing full benchmark for format: {framework}-NT{nt}", nt=nt, framework=framework):
                    with initialize(version_base=None, config_path="../configs"):
                        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                                       '++workload.workflow.generate_data=True',\
                                                                       f"++workload.framework={framework}", \
                                                                       f"++workload.reader.data_loader={framework}", \
                                                                       f"++workload.reader.read_threads={nt}", \
                                                                       'workload.train.computation_time=0.01', \
                                                                       'workload.evaluation.eval_time=0.005', \
                                                                       '++workload.train.epochs=1', \
                                                                       '++workload.dataset.num_files_train=16'])
                        benchmark = self.run_benchmark(cfg)
        self.clean()

    @pytest.mark.timeout(60, method="thread")
    def test_train(self) -> None:
        for fmt in "npz","jpeg", "png", "tfrecord", "hdf5":
                for framework in "tensorflow", "pytorch":
                    with self.subTest(f"Testing full benchmark for format: {fmt}-{framework}", fmt=fmt, framework=framework):
                        if fmt=="tfrecord" and framework=="pytorch":
                            continue
                        self.clean()
                        if (comm.rank==0):
                            logging.info("")
                            logging.info("="*80)
                            logging.info(f" DLIO training test for {fmt} format in {framework} framework")
                            logging.info("="*80)                        
                        with initialize(version_base=None, config_path="../configs"):
                            cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                                           '++workload.workflow.generate_data=True',\
                                                                           f"++workload.framework={framework}", \
                                                                           f"++workload.reader.data_loader={framework}", \
                                                                           f"++workload.dataset.format={fmt}", 
                                                                           'workload.train.computation_time=0.01', \
                                                                           'workload.evaluation.eval_time=0.005', \
                                                                           '++workload.train.epochs=1', \
                                                                           '++workload.dataset.num_files_train=16',\
                                                                           '++workload.reader.read_threads=1'])
                            benchmark=self.run_benchmark(cfg)                          
                        self.clean()

    @pytest.mark.timeout(60, method="thread")
    def test_custom_storage_root_train(self) -> None:
        storage_root="root_dir"
        for fmt in "npz","jpeg", "png", "tfrecord", "hdf5":
                for framework in "tensorflow", "pytorch":
                    with self.subTest(f"Testing full benchmark for format: {fmt}-{framework}", fmt=fmt, framework=framework):
                        if fmt=="tfrecord" and framework=="pytorch":
                            continue
                        self.clean(storage_root)
                        if (comm.rank==0):
                            logging.info("")
                            logging.info("="*80)
                            logging.info(f" DLIO training test for {fmt} format in {framework} framework")
                            logging.info("="*80)                        
                        with initialize(version_base=None, config_path="../configs"):
                            cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                                           '++workload.workflow.generate_data=True',\
                                                                           f"++workload.framework={framework}", \
                                                                           f"++workload.reader.data_loader={framework}", \
                                                                           f"++workload.dataset.format={fmt}",
                                                                           f"++workload.storage.storage_root={storage_root}", \
                                                                           'workload.train.computation_time=0.01', \
                                                                           'workload.evaluation.eval_time=0.005', \
                                                                           '++workload.train.epochs=1', \
                                                                           '++workload.dataset.num_files_train=16',\
                                                                           '++workload.reader.read_threads=1'])
                            benchmark=self.run_benchmark(cfg)                          
                        self.clean(storage_root)

if __name__ == '__main__':
    unittest.main()
