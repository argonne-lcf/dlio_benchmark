#!/usr/bin/env python
from collections import namedtuple

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

import os

from src.dlio_benchmark import DLIOBenchmark
import glob
class TestDLIOBenchmark(unittest.TestCase):
    def clean(self) -> None:
        comm.Barrier()
        if (comm.rank==0):
            shutil.rmtree("./checkpoints", ignore_errors=True)
            shutil.rmtree("./data/", ignore_errors=True)
            shutil.rmtree("./output", ignore_errors=True)
        comm.Barrier()
    def run_benchmark(self, cfg, verify=True):
        comm.Barrier()
        if (comm.rank==0):
            shutil.rmtree("./output", ignore_errors=True)
        comm.Barrier()            
        benchmark = DLIOBenchmark(cfg['workload'])
        benchmark.initialize()
        benchmark.run()
        benchmark.finalize()
        if (verify):
            self.assertEqual(len(glob.glob(benchmark.output_folder+"./*_load_and_proc_times.json")), benchmark.comm_size)
        return benchmark
    def test_gen_data(self) -> None:
        for fmt in "tfrecord", "jpeg", "png", "hdf5", "npz":
            with self.subTest(f"Testing data generator for format: {fmt}", fmt=fmt):
                self.clean()
                with initialize(version_base=None, config_path="../configs"):
                    cfg = compose(config_name='config', overrides=['++workload.workflow.train=False', \
                                                                   '++workload.workflow.generate_data=True',
                                                                   f"++workload.dataset.format={fmt}"])
                    benchmark=self.run_benchmark(cfg, verify=False)
                    if benchmark.args.num_subfolders_train<=1:
                        self.assertEqual(len(glob.glob(cfg.workload.dataset.data_folder + f"train/*.{fmt}")), cfg.workload.dataset.num_files_train)
                        self.assertEqual(len(glob.glob(cfg.workload.dataset.data_folder + f"valid/*.{fmt}")), cfg.workload.dataset.num_files_eval)
                    else:
                        self.assertEqual(len(glob.glob(cfg.workload.dataset.data_folder + f"train/*/*.{fmt}")), cfg.workload.dataset.num_files_train)
                        self.assertEqual(len(glob.glob(cfg.workload.dataset.data_folder + f"valid/*/*.{fmt}")), cfg.workload.dataset.num_files_eval)
        
    def test_iostat_profiling(self) -> None:
        self.clean()
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config', overrides=['++workload.workflow.train=False',
                                                           '++workload.workflow.generate_data=True'])

            benchmark=self.run_benchmark(cfg, verify=False)
            cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=True',
                                                           'workload.train.computation_time=0.01',
                                                           'workload.evaluation.eval_time=0.005',
                                                           'workload.train.epochs=1',
                                                           'workload.workflow.profiling=True',
                                                           'workload.profiling.profiler=iostat'])
            benchmark=self.run_benchmark(cfg)
            assert(os.path.isfile(benchmark.output_folder+"/iostat.json"))
            if (comm.rank==0):
                os.makedirs(benchmark.output_folder+"/.hydra/", exist_ok=True)
                yl: str = OmegaConf.to_yaml(cfg)
                with open(benchmark.output_folder+"./.hydra/config.yaml", "w") as f:
                    OmegaConf.save(cfg, f)
                with open(benchmark.output_folder+"./.hydra/overrides.yaml", "w") as f:
                    f.write('[]')
                os.system(f"python src/dlio_postprocessor.py --output-folder={benchmark.output_folder}")
    def test_checkpoint_epoch(self) -> None:
        self.clean()
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
    def test_checkpoint_step(self) -> None:
        self.clean()
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
            nstep = dataset.num_files_train * dataset.num_samples_per_file // dataset.batch_size//benchmark.comm_size
            ncheckpoints=nstep//2*8
            self.assertEqual(len(glob.glob("./checkpoints/*.bin")), ncheckpoints)
    def test_eval(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.workflow.train=True',\
                                     '++workload.workflow.generate_data=True', \
                                     'workload.train.computation_time=0.01', \
                                     'workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=4', '++workload.workflow.evaluation=True'])
            benchmark=self.run_benchmark(cfg)
    def test_multi_threads(self) -> None:
        self.clean()
        for framework in "tensorflow", "pytorch":
            for nt in 1, 2, 4:
                with self.subTest(f"Testing full benchmark for format: {framework}-NT{nt}", nt=nt, framework=framework):
                    with initialize(version_base=None, config_path="../configs"):
                        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                                       '++workload.workflow.generate_data=True',\
                                                                       f"++workload.framework={framework}", \
                                                                       f"++workload.data_reader.data_loader={framework}", \
                                                                       f"++workload.data_reader.read_threads={nt}", \
                                                                       'workload.train.computation_time=0.01', \
                                                                       'workload.evaluation.eval_time=0.005', \
                                                                       '++workload.train.epochs=1', \
                                                                       '++workload.dataset.num_files_train=16'])
                        benchmark = self.run_benchmark(cfg)

    def test_train(self) -> None:
        for fmt in "npz","jpeg", "png", "tfrecord", "hdf5":
                for framework in "tensorflow", "pytorch":
                    self.clean()
                    with self.subTest(f"Testing full benchmark for format: {fmt}-{framework}", fmt=fmt, framework=framework):
                        if fmt=="tfrecord" and framework=="pytorch":
                            continue
                        
                        with initialize(version_base=None, config_path="../configs"):
                            cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                                           '++workload.workflow.generate_data=True',\
                                                                           f"++workload.framework={framework}", \
                                                                           f"++workload.data_reader.data_loader={framework}", \
                                                                           f"++workload.dataset.format={fmt}", 
                                                                           'workload.train.computation_time=0.01', \
                                                                           'workload.evaluation.eval_time=0.005', \
                                                                           '++workload.train.epochs=1', \
                                                                           '++workload.dataset.num_files_train=16',\
                                                                           '++workload.data_reader.read_threads=1'])
                            benchmark=self.run_benchmark(cfg)
if __name__ == '__main__':
    unittest.main()
