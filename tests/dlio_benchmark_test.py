#!/usr/bin/env python
from collections import namedtuple

from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import hydra
import unittest
import yaml

from omegaconf import OmegaConf

import os

from src.dlio_benchmark import DLIOBenchmark
import glob
class TestDLIOBenchmark(unittest.TestCase):
    def test_step0_gen_data(self) -> None:
        with initialize(version_base=None, config_path="./configs"):
            cfg = compose(config_name='config', overrides=['++workload.workflow.train=False', '++workload.workflow.generate_data=True'])
            
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()
            benchmark.run()
            benchmark.finalize()
            assert(len(glob.glob(cfg.workload.dataset.data_folder + "train/*.npz"))==cfg.workload.dataset.num_files_train)
            assert(len(glob.glob(cfg.workload.dataset.data_folder + "valid/*.npz"))==cfg.workload.dataset.num_files_eval)
        return 0
    def test_step1_train(self) -> None:
        with initialize(version_base=None, config_path="./configs"):
            cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', '++workload.workflow.generate_data=False', 'workload.train.computation_time=0.01', 'workload.evaluation.eval_time=0.005', 'workload.train.epochs=1'])
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()
            benchmark.run()
            benchmark.finalize()
            os.makedirs(benchmark.output_folder+"/.hydra/", exist_ok=True)
            yl: str = OmegaConf.to_yaml(cfg)
            with open(benchmark.output_folder+"./.hydra/config.yaml", "w") as f:
                OmegaConf.save(cfg, f)
            with open(benchmark.output_folder+"./.hydra/overrides.yaml", "w") as f:
                f.write('[]')
            assert(len(glob.glob(benchmark.output_folder+"./*_load_and_proc_times.json"))==benchmark.comm_size)

if __name__ == '__main__':
    unittest.main()
