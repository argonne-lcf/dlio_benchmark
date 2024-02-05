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
import os
import math
import hydra
import logging
import pandas as pd
from time import time, sleep
import json
import numpy as np

# Reduce TF and CUDA logging
from numpy import random

from dlio_benchmark.checkpointing.checkpointing_factory import CheckpointingFactory
from dlio_benchmark.common.constants import MODULE_DLIO_BENCHMARK

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# Remove PyTorch warning when libtorch_cuda_cu.so isn't found
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dataclasses import dataclass
from dlio_benchmark.utils.utility import utcnow, measure_performance, get_trace_name, DLIOMPI
from omegaconf import DictConfig, OmegaConf
from dlio_benchmark.utils.statscounter import StatsCounter
from hydra.core.config_store import ConfigStore
from dlio_benchmark.utils.config import LoadConfig, ConfigArguments
from dlio_benchmark.common.enumerations import Profiler, DatasetType, StorageType, MetadataType
from dlio_benchmark.profiler.profiler_factory import ProfilerFactory
from dlio_benchmark.framework.framework_factory import FrameworkFactory
from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_profiler.logger import dlio_logger as PerfTrace, fn_interceptor as Profile

dlp = Profile(MODULE_DLIO_BENCHMARK)


class DLIOBenchmark(object):
    """
    The Benchmark represents the I/O behavior of deep learning applications.
    """

    def __init__(self, cfg):
        """
        This initializes the DLIO benchmark. Intialization includes:
        <ul>
            <li> argument parser </li>
            <li> profiler instances </li>
            <li> internal components </li>
            <li> local variables </li>
        </ul>
        """
        t0 = time()
        self.args = ConfigArguments.get_instance()
        LoadConfig(self.args, cfg)
        self.storage = StorageFactory().get_storage(self.args.storage_type, self.args.storage_root,
                                                    self.args.framework)

        self.output_folder = self.args.output_folder
        os.makedirs(self.args.output_folder, mode=0o755, exist_ok=True)
        self.comm = DLIOMPI.get_instance().comm()
        self.my_rank = self.args.my_rank = DLIOMPI.get_instance().rank()
        self.comm_size = self.args.comm_size = DLIOMPI.get_instance().size()
        self.data_folder = self.args.data_folder
        self.storage_root = self.args.storage_root
        if self.args.storage_root:
            self.storage.create_namespace(exist_ok=True)
        self.framework = FrameworkFactory().get_framework(self.args.framework,
                                                          self.args.do_profiling)

        # Delete previous logfile
        if self.my_rank == 0:
            if os.path.isfile(self.args.logfile_path):
                os.remove(self.args.logfile_path)
        self.comm.barrier()
        # Configure the logging library
        self.args.configure_dlio_logging(is_child=False)
        self.dlio_profiler = self.args.configure_dlio_profiler(is_child=False, use_pid=False)
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_DLIO_BENCHMARK):
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Running DLIO with {self.args.comm_size} process(es)")
                try:
                    logging.info(
                        f"{utcnow()} Reading workload YAML config file '{hydra_cfg.runtime.config_sources[1]['path']}/workload/{hydra_cfg.runtime.choices.workload}.yaml'")
                except:
                    pass

            self.generate_only = self.args.generate_only
            self.do_profiling = self.args.do_profiling

            self.data_generator = None
            self.num_files_train = self.args.num_files_train
            self.num_subfolders_train = self.args.num_subfolders_train
            self.num_subfolders_eval = self.args.num_subfolders_eval
            self.num_samples = self.args.num_samples_per_file
            self.total_training_steps = self.args.total_training_steps

            self.epochs = self.args.epochs
            self.batch_size = self.args.batch_size
            self.computation_time = self.args.computation_time
            self.computation_time_stdev = self.args.computation_time_stdev

            if self.do_profiling:
                self.profiler = ProfilerFactory().get_profiler(self.args.profiler)

            if self.args.generate_data:
                self.data_generator = GeneratorFactory.get_generator(self.args.format)

            # Checkpointing support
            self.do_checkpoint = self.args.do_checkpoint
            self.steps_between_checkpoints = self.args.steps_between_checkpoints
            self.epochs_between_checkpoints = self.args.epochs_between_checkpoints
            self.checkpoint_after_epoch = self.args.checkpoint_after_epoch

            # Evaluation support
            self.do_eval = self.args.do_eval
            self.num_files_eval = self.args.num_files_eval

            self.batch_size_eval = self.args.batch_size_eval
            self.eval_time = self.args.eval_time
            self.eval_time_stdev = self.args.eval_time_stdev
            self.eval_after_epoch = self.args.eval_after_epoch
            self.epochs_between_evals = self.args.epochs_between_evals
        self.stats = StatsCounter()

    @dlp.log
    def initialize(self):
        """
        Initializes the benchmark runtime.
        - It generates the required data
        - Start profiling session for Darshan and Tensorboard.
        """
        self.comm.barrier()
        if self.args.debug and self.args.my_rank == 0:
            input("Debug mode: Press enter to start\n")

        if self.args.generate_data:
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Starting data generation")
            self.data_generator.generate()
            # important to have this barrier to ensure that the data generation is done for all the ranks
            self.comm.barrier()
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Generation done")

        if not self.generate_only and self.do_profiling:
            self.profiler.start()
            self.framework.start_framework_profiler()
            self.comm.barrier()
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Profiling Started with {self.args.profiler}")
        self.comm.barrier()
        file_list_train = []
        file_list_eval = []
        num_subfolders = 0
        for dataset_type in [DatasetType.TRAIN, DatasetType.VALID]:
            if dataset_type == DatasetType.TRAIN:
                num_subfolders = self.num_subfolders_train
            else:
                num_subfolders = self.num_subfolders_eval
            filenames = self.storage.walk_node(os.path.join(self.args.data_folder, f"{dataset_type}"))
            if (len(filenames) == 0):
                continue
            if self.storage.get_node(
                    os.path.join(self.args.data_folder, f"{dataset_type}",
                                 filenames[0])) == MetadataType.DIRECTORY:
                assert (num_subfolders == len(filenames))
                fullpaths = self.storage.walk_node(
                    os.path.join(self.args.data_folder, f"{dataset_type}/*/*.{self.args.format}"),
                    use_pattern=True)
                files = [self.storage.get_basename(f) for f in fullpaths]
                idx = np.argsort(files)
                fullpaths = [fullpaths[i] for i in idx]
            else:
                assert (num_subfolders == 0)
                fullpaths = [self.storage.get_uri(os.path.join(self.args.data_folder, f"{dataset_type}", entry))
                             for entry in filenames if entry.endswith(f'{self.args.format}')]
                fullpaths = sorted(fullpaths)
            logging.debug(f"subfolder {num_subfolders} fullpaths {fullpaths}")
            if dataset_type is DatasetType.TRAIN:
                file_list_train = fullpaths
            elif dataset_type is DatasetType.VALID:
                file_list_eval = fullpaths
        if not self.generate_only and self.num_files_train > len(file_list_train):
            raise Exception(
                "Not enough training dataset is found; Please run the code with ++workload.workflow.generate_data=True")
        if self.do_eval and self.num_files_eval > len(file_list_eval):
            raise Exception(
                "Not enough evaluation dataset is found; Please run the code with ++workload.workflow.generate_data=True")
        if (self.num_files_train < len(file_list_train)):
            logging.warning(
                f"Number of files for training in {os.path.join(self.args.data_folder, f'{DatasetType.TRAIN}')} ({len(file_list_train)}) is more than requested ({self.num_files_train}). A subset of files will be used ")
            file_list_train = file_list_train[:self.num_files_train]
        if (self.num_files_eval < len(file_list_eval)):
            logging.warning(
                f"Number of files for evaluation in {os.path.join(self.args.data_folder, f'{DatasetType.VALID}')} ({len(file_list_eval)}) is more than requested ({self.num_files_eval}). A subset of files will be used ")
            file_list_eval = file_list_eval[:self.num_files_eval]
        self.args.derive_configurations(file_list_train, file_list_eval)
        self.checkpointing_mechanism = CheckpointingFactory().get_mechanism(self.args.checkpoint_mechanism)
        self.args.validate()
        self.comm.barrier()

    @dlp.log
    def _eval(self, epoch):
        """
        Evaluation loop will read a separate dataset and has its own own computation time.
        """
        self.args.reconfigure(epoch, DatasetType.VALID)
        step = 1
        total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
        loader = self.framework.get_loader(DatasetType.VALID)
        loader.read()
        t0 = time()
        for batch in dlp.iter(loader.next()):
            self.stats.eval_batch_loaded(epoch, step, t0)
            eval_time = 0.0
            if self.eval_time > 0:
                if self.eval_time_stdev > 0:
                    eval_time = random.normal(self.eval_time, self.eval_time_stdev)
                else:
                    eval_time = self.eval_time
                self.framework.compute(batch, epoch, step, eval_time)
            self.stats.eval_batch_processed(epoch, step, t0, eval_time)

            step += 1
            if step > total:
                break
            t0 = time()
        return step - 1

    @dlp.log
    def _train(self, epoch):
        """
        Training loop for reading the dataset and performing training computations.
        :return: returns total steps.
        """
        block = 1  # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1  # Steps are taken within blocks
        max_steps = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
        self.steps_per_epoch = max_steps
        # Start the very first block
        self.stats.start_block(epoch, block)

        loader = self.framework.get_loader(dataset_type=DatasetType.TRAIN)
        t0 = time()
        for batch in dlp.iter(loader.next()):
            self.stats.batch_loaded(epoch, overall_step, block, t0)

            # Log a new block, unless it's the first one which we've already logged before the loop
            if block_step == 1 and block != 1:
                self.stats.start_block(epoch, block)
            computation_time = 0.0
            if self.computation_time > 0:
                self.framework.trace_object("Train", overall_step, 1)
                if self.computation_time_stdev > 0:
                    computation_time = random.normal(self.computation_time, self.computation_time_stdev)
                else:
                    computation_time = self.computation_time
                self.framework.compute(batch, epoch, block_step, computation_time)
            self.comm.barrier()
            self.stats.batch_processed(epoch, overall_step, block, t0, computation_time)
            if self.do_checkpoint and (
                    self.steps_between_checkpoints >= 0) and overall_step == self.next_checkpoint_step:
                self.stats.end_block(epoch, block, block_step)
                self.stats.start_ckpt(epoch, block, overall_step)
                self.checkpointing_mechanism.checkpoint(epoch, overall_step)
                self.stats.end_ckpt(epoch, block)
                block += 1
                # Reset the number of steps after every checkpoint to mark the start of a new block
                block_step = 1
                self.next_checkpoint_step += self.steps_between_checkpoints
            else:
                block_step += 1

            if overall_step >= max_steps or overall_step == self.total_training_steps:
                if self.args.my_rank == 0:
                    logging.info(f"{utcnow()} Maximum number of steps reached")
                if (block_step != 1 and self.do_checkpoint) or (not self.do_checkpoint):
                    self.stats.end_block(epoch, block, block_step - 1)
                break
            overall_step += 1
            t0 = time()
        self.comm.barrier()
        if self.do_checkpoint and (self.steps_between_checkpoints < 0) and (epoch == self.next_checkpoint_epoch):
            self.stats.end_block(epoch, block, block_step)
            self.stats.start_ckpt(epoch, block, overall_step)
            self.checkpointing_mechanism.checkpoint(epoch, overall_step)
            self.stats.end_ckpt(epoch, block)
            self.next_checkpoint_epoch += self.epochs_between_checkpoints
        self.comm.barrier()
        return overall_step

    @dlp.log
    def run(self):
        """
        Run the total epochs for training. 
        On each epoch, it prepares dataset for reading, it trains, and finalizes the dataset.
        If evaluation is enabled, it reads the eval dataset, performs evaluation and finalizes.
        """
        self.stats.start_run()
        if not self.generate_only:
            # Print out the expected number of steps for each epoch and evaluation
            if self.my_rank == 0:
                total = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
                logging.info(
                    f"{utcnow()} Max steps per epoch: {total} = {self.num_samples} * {self.num_files_train} / {self.batch_size} / {self.comm_size} (samples per file * num files / batch size / comm size)")

                if self.do_eval:
                    total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
                    logging.info(
                        f"{utcnow()} Steps per eval: {total} = {self.num_samples} * {self.num_files_eval} / {self.batch_size_eval} / {self.comm_size} (samples per file * num files / batch size eval / comm size)")

            # Keep track of the next epoch at which we will evaluate
            next_eval_epoch = self.eval_after_epoch
            self.next_checkpoint_epoch = self.checkpoint_after_epoch
            epoch = 1
            self.args.reconfigure(epoch, DatasetType.TRAIN)
            # Initialize the dataset
            self.framework.init_loader(self.args.format, epoch=epoch, data_loader=self.args.data_loader)
            loader = self.framework.get_loader(dataset_type=DatasetType.TRAIN)
            loader.read()
            for epoch in range(1, self.epochs + 1):
                self.next_checkpoint_step = self.steps_between_checkpoints
                self.stats.start_train(epoch)
                steps = self._train(epoch)
                self.stats.end_train(epoch, steps)
                logging.debug(f"{utcnow()} Rank {self.my_rank} returned after {steps} steps.")
                self.framework.get_loader(DatasetType.TRAIN).finalize()
                # Perform evaluation if enabled
                if self.do_eval and epoch >= next_eval_epoch:
                    next_eval_epoch += self.epochs_between_evals

                    self.stats.start_eval(epoch)
                    self._eval(epoch)
                    self.stats.end_eval(epoch)
                    self.framework.get_loader(DatasetType.VALID).finalize()
        self.stats.end_run()

    @dlp.log
    def finalize(self):
        """
        It finalizes the dataset once training is completed.
        """
        self.comm.barrier()
        self.checkpointing_mechanism.finalize()
        if not self.generate_only:
            if self.do_profiling:
                self.profiler.stop()
                self.framework.stop_framework_profiler()
                self.comm.barrier()
                if self.my_rank == 0:
                    logging.info(f"{utcnow()} Profiling stopped")
            if not self.args.keep_files:
                logging.info(f"{utcnow()} Keep files set to False. Deleting dataset")
                self.comm.barrier()
                if self.my_rank == 0:
                    if self.storage.get_node(self.args.data_folder):
                        self.storage.delete_node(self.args.data_folder)
                        logging.info(f"{utcnow()} Deleted data files")

            # Save collected stats to disk
            self.stats.finalize()
            self.stats.save_data()
        self.comm.barrier()
        self.args.finalize_dlio_profiler(self.dlio_profiler)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    """
    The main method to start the benchmark runtime.
    """
    DLIOMPI.get_instance().initialize()
    benchmark = DLIOBenchmark(cfg['workload'])
    os.environ["DARSHAN_DISABLE"] = "1"
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    DLIOMPI.get_instance().finalize()

if __name__ == '__main__':
    main()
    exit(0)
