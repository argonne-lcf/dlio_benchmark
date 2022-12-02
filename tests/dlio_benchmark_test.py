#!/usr/bin/env python
from collections import namedtuple

from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import hydra
import unittest
import yaml
import shutil

import pytest

import os

from src.dlio_benchmark import DLIOBenchmark
import glob
class TestDLIOBenchmark(unittest.TestCase):
    def clean(self) -> None:
        shutil.rmtree("./checkpoints", ignore_errors=True)
        shutil.rmtree("./data/", ignore_errors=True)
        shutil.rmtree("./output", ignore_errors=True)
    def run_benchmark(self, cfg):
        shutil.rmtree("./output", ignore_errors=True)
        benchmark = DLIOBenchmark(cfg['workload'])
        benchmark.initialize()
        benchmark.run()
        benchmark.finalize()
        return benchmark
    def test0_gen_data1_formats(self) -> None:
        for fmt in "tfrecord", "jpeg", "png", "npz":
            with self.subTest(f"Testing data generator for format: {fmt}", fmt=fmt):
                self.clean()
                with initialize(version_base=None, config_path="../configs"):
                    cfg = compose(config_name='config', overrides=['++workload.workflow.train=False', \
                                                                   '++workload.workflow.generate_data=True',
                                                                   f"++workload.dataset.format={fmt}"])
                    benchmark=self.run_benchmark(cfg)
                    if benchmark.args.num_subfolders_train<=1:
                        self.assertEqual(len(glob.glob(cfg.workload.dataset.data_folder + f"train/*.{fmt}")), cfg.workload.dataset.num_files_train)
                        self.assertEqual(len(glob.glob(cfg.workload.dataset.data_folder + f"valid/*.{fmt}")), cfg.workload.dataset.num_files_eval)
                    else:
                        self.assertEqual(len(glob.glob(cfg.workload.dataset.data_folder + f"train/*/*.{fmt}")), cfg.workload.dataset.num_files_train)
                        self.assertEqual(len(glob.glob(cfg.workload.dataset.data_folder + f"valid/*/*.{fmt}")), cfg.workload.dataset.num_files_eval)
        
    def test1_train(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', '++workload.workflow.generate_data=False', 'workload.train.computation_time=0.01', 'workload.evaluation.eval_time=0.005', 'workload.train.epochs=1'])
            benchmark=self.run_benchmark(cfg)            

            os.makedirs(benchmark.output_folder+"/.hydra/", exist_ok=True)
            yl: str = OmegaConf.to_yaml(cfg)
            with open(benchmark.output_folder+"./.hydra/config.yaml", "w") as f:
                OmegaConf.save(cfg, f)
            with open(benchmark.output_folder+"./.hydra/overrides.yaml", "w") as f:
                f.write('[]')
            self.assertEqual(len(glob.glob(benchmark.output_folder+"./*_load_and_proc_times.json")), benchmark.comm_size)
    def test2_checkpoint1_epoch(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.workflow.train=True',\
                                     '++workload.workflow.generate_data=False', \
                                     '++workload.train.computation_time=0.01', \
                                     '++workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=8', '++workload.workflow.checkpoint=True', \
                                     '++workload.checkpoint.epochs_between_checkpoints=2'])
            shutil.rmtree("./checkpoints", ignore_errors=True)                                    
            benchmark=self.run_benchmark(cfg)            
            self.assertEqual(len(glob.glob("./checkpoints/*.bin")), 4)
    def test2_checkpoint2_step(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.workflow.train=True',\
                                     '++workload.workflow.generate_data=False', \
                                     '++workload.train.computation_time=0.01', \
                                     '++workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=8', '++workload.workflow.checkpoint=True', \
                                     '++workload.checkpoint.steps_between_checkpoints=2'])
            benchmark = DLIOBenchmark(cfg['workload'])
            dataset = cfg['workload']['dataset']
            nstep = dataset.num_files_train * dataset.num_samples_per_file // dataset.batch_size//benchmark.comm_size
            ncheckpoints=nstep//2*8
            shutil.rmtree("./checkpoints", ignore_errors=True)                        
            benchmark=self.run_benchmark(cfg)                        
            self.assertEqual(len(glob.glob("./checkpoints/*.bin")), ncheckpoints)
    def test3_eval(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.workflow.train=True',\
                                     '++workload.workflow.generate_data=False', \
                                     'workload.train.computation_time=0.01', \
                                     'workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=4', '++workload.workflow.evaluation=True'])
            benchmark=self.run_benchmark(cfg)
            self.assertEqual(len(glob.glob(benchmark.output_folder+"./*_load_and_proc_times.json")), benchmark.comm_size)            
    def test2_tf_loader(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.data_reader.data_loader=tensorflow',\
                                     '++workload.framework=tensorflow', \
                                     '++workload.workflow.generate_data=False', \
                                     'workload.train.computation_time=0.01', \
                                     'workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=4', '++workload.workflow.evaluation=True'])
            benchmark = self.run_benchmark(cfg)                                                
            self.assertEqual(len(glob.glob(benchmark.output_folder+"./*_load_and_proc_times.json")), benchmark.comm_size)            
    def test2_torch_loader(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name='config',
                          overrides=['++workload.data_reader.data_loader=pytorch',\
                                     '++workload.framework=pytorch', \
                                     '++workload.workflow.generate_data=False', \
                                     'workload.train.computation_time=0.01', \
                                     'workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=4', '++workload.workflow.evaluation=True'])
            benchmark = self.run_benchmark(cfg)
            self.assertEqual(len(glob.glob(benchmark.output_folder+"./*_load_and_proc_times.json")), benchmark.comm_size)            
    def test_full_formats(self) -> None:
        for fmt in "tfrecord", "jpeg", "png", "npz":
            with self.subTest(f"Testing full benchmark for format: {fmt}", fmt=fmt):
                for framework in "tensorflow", "pytorch":
                    if fmt=="tfrecord" and framework=="pytorch":
                        pass
                    self.clean()
                    with initialize(version_base=None, config_path="../configs"):
                        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                                       '++workload.workflow.generate_data=True',\
                                                                       f"++workload.framework={framework}", \
                                                                       f"++workload.data_reader.data_loader={framework}", \
                                                                       f"++workload.dataset.format={fmt}", 
                                                                       'workload.train.computation_time=0.01', \
                                                                       'workload.evaluation.eval_time=0.005', \
                                                                       '++workload.train.epochs=1', \
                                                                       '++workload.dataset.num_files_train=4'])
                        benchmark=self.run_benchmark(cfg)
                        self.assertEqual(len(glob.glob(benchmark.output_folder+"./*_load_and_proc_times.json")), benchmark.comm_size)
if __name__ == '__main__':
    unittest.main()
