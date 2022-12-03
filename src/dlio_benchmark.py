"""
   Copyright © 2022, UChicago Argonne, LLC
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
import shutil
import logging
import pandas as pd
from time import time

# Reduce TF and CUDA logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Remove PyTorch warning when libtorch_cuda_cu.so isn't found
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dataclasses import dataclass
from src.utils.utility import utcnow
from omegaconf import DictConfig, OmegaConf
from src.utils.statscounter import StatsCounter
from hydra.core.config_store import ConfigStore
from src.utils.config import LoadConfig, ConfigArguments
from src.common.enumerations import Profiler, DatasetType
from src.profiler.profiler_factory import ProfilerFactory
from src.framework.framework_factory import FrameworkFactory
from src.data_generator.generator_factory import GeneratorFactory


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
        self.args = ConfigArguments.get_instance()
        

        LoadConfig(self.args, cfg)
        self.args.validate()
        try:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            self.args.output_folder = hydra_cfg['runtime']['output_dir']
        except:
            self.args.output_folder = 'output/'

        self.output_folder = self.args.output_folder
        
        self.logfile = os.path.join(self.output_folder, self.args.log_file)

        self.data_folder = self.args.data_folder
        self.output_folder = self.args.output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)

        self.framework = FrameworkFactory().get_framework(self.args.framework,
                                                          self.args.do_profiling)

        self.my_rank = self.args.my_rank = self.framework.rank()
        self.comm_size = self.args.comm_size = self.framework.size()


        # Delete previous logfile
        if self.my_rank == 0:
            if os.path.isfile(self.logfile):
                os.remove(self.logfile)
        self.framework.barrier()
        # Configure the logging library
        log_level = logging.DEBUG if self.args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            handlers=[
                logging.FileHandler(self.logfile, mode = "a", encoding='utf-8'),
                logging.StreamHandler()
            ],
            format='[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'  # logging's max timestamp resolution is msecs, we will pass in usecs in the message
        )
 

        if self.args.my_rank==0:
            logging.info(f"{utcnow()} Running DLIO with {self.args.comm_size} process(es)")
            try:
                logging.info(f"{utcnow()} Reading YAML config file './configs/workload/{hydra_cfg.runtime.choices.workload}.yaml'" )
            except:
                pass

        self.generate_only = self.args.generate_only
        self.do_profiling = self.args.do_profiling

        self.data_generator = None
        self.num_files_train = self.args.num_files_train
        self.num_samples = self.args.num_samples_per_file
        self.total_training_steps = self.args.total_training_steps
        
        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.computation_time = self.args.computation_time

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
        self.eval_after_epoch = self.args.eval_after_epoch
        self.epochs_between_evals = self.args.epochs_between_evals

        # Hold various lists/dicts for statistics
        self.time_to_load_train_batch = []
        self.time_to_process_train_batch = []

        self.time_to_load_eval_batch = []
        self.time_to_process_eval_batch = []

        self.epoch_time_ranges = []
        self.eval_time_ranges = []

        self.ckpt_time_ranges = []

        # Indexed by epoch number, contains start-end timestamps and other information
        self.per_epoch_stats = {}
        self.stats = StatsCounter()

    def initialize(self):
        """
        Initializes the benchmark runtime.
        - It generates the required data
        - Start profiling session for Darshan and Tensorboard.
        """
        self.framework.barrier()
        if self.args.debug and self.args.my_rank == 0:
            input("Debug mode: Press enter to start\n")

        if self.args.generate_data:
            if self.args.my_rank==0:
                logging.info(f"{utcnow()} Starting data generation")
            self.data_generator.generate()
            # important to have this barrier to ensure that the data generation is done for all the ranks
            self.framework.barrier()
            if self.args.my_rank==0:
                logging.info(f"{utcnow()} Generation done")

        if not self.generate_only and self.do_profiling:
            self.profiler.start()
            self.framework.start_framework_profiler()
            self.framework.barrier()
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Profiling Started with {self.args.profiler}")
        self.framework.init_reader(self.args.format, self.args.data_loader)
        self.framework.barrier()

    def _eval(self, epoch):
        """
        Evaluation loop will read a separate dataset and has its own own computation time.
        """
        step = 1
        total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
        t0 = time() 
        for batch in self.framework.get_reader(DatasetType.VALID).next():
            self.stats.eval_batch_loaded(epoch, step, t0)

            if self.eval_time > 0:
                self.framework.compute(epoch, step, self.eval_time)

            self.stats.eval_batch_processed(epoch, step, t0)

            step += 1
            if step > total:
                return step - 1
                
            self.framework.barrier()
            t0 = time()

        return step - 1

    def _train(self, epoch):
        """
        Training loop for reading the dataset and performing training computations.
        :return: returns total steps.
        """
        block = 1   # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1   # Steps are taken within blocks
        max_steps = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)

        # Start the very first block
        self.stats.start_block(epoch, block)
        t0 = time()
        for batch in self.framework.get_reader(dataset_type=DatasetType.TRAIN).next():
            self.stats.batch_loaded(epoch, overall_step, block, t0)
            self.framework.barrier()
            # Log a new block, unless it's the first one which we've already logged before the loop
            if block_step == 1 and block != 1:
                self.stats.start_block(epoch, block)
            
            if self.computation_time > 0:
                self.framework.compute(epoch, block_step, self.computation_time)
            self.framework.barrier()

            self.stats.batch_processed(epoch, overall_step, block, t0)


            if self.do_checkpoint and (self.steps_between_checkpoints>=0) and overall_step == self.next_checkpoint_step:
                self.stats.end_block(epoch, block, block_step)
                self.stats.start_ckpt(epoch, block, overall_step)
                self.framework.checkpoint(epoch, overall_step)
                self.stats.end_ckpt(epoch, block)
                self.framework.barrier()
                block += 1
                # Reset the number of steps after every checkpoint to mark the start of a new block
                block_step = 1
                self.next_checkpoint_step += self.steps_between_checkpoints
            else:
                block_step += 1

            if overall_step >= max_steps or overall_step == self.total_training_steps:
                self.framework.barrier()
                if self.args.my_rank==0:
                    logging.info(f"{utcnow()} Maximum number of steps reached")
                if (block_step!=1 and self.do_checkpoint) or (not self.do_checkpoint):
                    self.stats.end_block(epoch, block, block_step-1)
                break
                
            overall_step += 1
            t0 = time()


        if self.do_checkpoint and (self.steps_between_checkpoints < 0) and (epoch == self.next_checkpoint_epoch):
            self.stats.end_block(epoch, block, block_step)
            self.stats.start_ckpt(epoch, block, overall_step)
            self.framework.checkpoint(epoch, overall_step)
            self.stats.end_ckpt(epoch, block)
            self.framework.barrier()
            self.next_checkpoint_epoch += self.epochs_between_checkpoints
        return overall_step

    
    def run(self):
        """
        Run the total epochs for training. 
        On each epoch, it prepares dataset for reading, it trains, and finalizes the dataset.
        If evaluation is enabled, it reads the eval dataset, performs evaluation and finalizes.
        """
        if not self.generate_only:
            # Print out the expected number of steps for each epoch and evaluation
            if self.my_rank == 0:
                total = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
                logging.info(f"{utcnow()} Max steps per epoch: {total} = {self.num_samples} * {self.num_files_train} / {self.batch_size} / {self.comm_size} (samples per file * num files / batch size / comm size)")

                if self.do_eval:
                    total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
                    logging.info(f"{utcnow()} Steps per eval: {total} = {self.num_samples} * {self.num_files_eval} / {self.batch_size_eval} / {self.comm_size} (samples per file * num files / batch size eval / comm size)")
            
            # Keep track of the next epoch at which we will evaluate
            next_eval_epoch = self.eval_after_epoch
            self.next_checkpoint_epoch = self.checkpoint_after_epoch

            for epoch in range(1, self.epochs + 1):
                self.next_checkpoint_step = self.steps_between_checkpoints                
                self.stats.start_epoch(epoch)

                # Initialize the dataset
                self.framework.get_reader(dataset_type=DatasetType.TRAIN).read(epoch)
                self.framework.barrier()

                steps = self._train(epoch)
                self.stats.end_epoch(epoch, steps)
                logging.debug(f"{utcnow()} Rank {self.my_rank} returned after {steps} steps.")



                self.framework.barrier()
                self.framework.get_reader(DatasetType.TRAIN).finalize()

                # Perform evaluation if enabled
                if self.do_eval and epoch >= next_eval_epoch:
                    next_eval_epoch += self.epochs_between_evals

                    self.stats.start_eval(epoch)
                
                    # Initialize the eval dataset
                    self.framework.get_reader(DatasetType.VALID).read(epoch)
                    self.framework.barrier()
                    
                    self._eval(epoch)
                    self.stats.end_eval(epoch)

                    self.framework.barrier()
                    self.framework.get_reader(DatasetType.VALID).finalize()

    def finalize(self):
        """
        It finalizes the dataset once training is completed.
        """
        self.framework.barrier()
        if not self.generate_only:
            if self.do_profiling:
                self.profiler.stop()
                self.framework.stop_framework_profiler()
                self.framework.barrier()
                if self.my_rank == 0:
                    logging.info(f"{utcnow()} Profiling stopped")
            if not self.args.keep_files:
                logging.info(f"{utcnow()} Keep files set to False. Deleting dataset")
                self.framework.barrier()
                if self.my_rank == 0:
                    if os.path.exists(self.args.data_folder):
                        shutil.rmtree(self.args.data_folder)
                        logging.info(f"{utcnow()} Deleted data files")
            
            # Save collected stats to disk
            self.stats.save_data()
        if self.my_rank==0:
            logging.info(f"{utcnow()} Saved outputs in {self.output_folder}")
        self.framework.barrier()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg : DictConfig) -> None:
    """
    The main method to start the benchmark runtime.
    """
    os.environ["DARSHAN_DISABLE"] = "1"
    benchmark = DLIOBenchmark(cfg['workload'])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()

if __name__ == '__main__':
    main()
    exit(0)
