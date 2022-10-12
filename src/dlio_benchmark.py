"""
   Copyright 2021 UChicago Argonne, LLC

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
from time import time

from src.common.enumerations import Profiler
from src.data_generator.generator_factory import GeneratorFactory
from src.framework.framework_factory import FrameworkFactory
from src.profiler.profiler_factory import ProfilerFactory
from src.utils.argument_parser import ArgumentParser
from src.utils.utility import utcnow

import math
import shutil
import os
import logging

# Remove (some) TF and CUDA logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DLIOBenchmark(object):
    """
    The Benchmark represents the I/O behavior of deep learning applications.
    """

    def __init__(self):
        """
        This initializes the DLIO benchmark. Intialization includes:
        - argument parser
        - profiler instances
        - internal components
        - local variables
        """
        self.arg_parser = ArgumentParser.get_instance()
        self.output_folder = self.arg_parser.args.output_folder

        log_level = logging.DEBUG if self.arg_parser.args.debug else logging.INFO

        logging.basicConfig(
            level=log_level,
            handlers=[
                logging.FileHandler(os.path.join(self.output_folder, self.arg_parser.args.log_file), mode = "a", encoding='utf-8'),
                logging.StreamHandler()
            ],
            format='%(message)s [%(pathname)s:%(lineno)d]'  # logging's max timestamp resolution is msecs, we will pass in usecs in the message
        )

        self.framework = FrameworkFactory().get_framework(self.arg_parser.args.framework,
                                                          self.arg_parser.args.profiling)
        self.my_rank = self.arg_parser.args.my_rank = self.framework.rank()
        self.comm_size = self.arg_parser.args.comm_size = self.framework.size()
        self.framework.init_reader(self.arg_parser.args.format)
        self.darshan = None
        self.data_generator = None
        self.num_files_train = self.arg_parser.args.num_files_train
        self.num_samples = self.arg_parser.args.num_samples
        self.batch_size = self.arg_parser.args.batch_size
        self.computation_time = self.arg_parser.args.computation_time

        if self.arg_parser.args.profiling:
            self.darshan = ProfilerFactory().get_profiler(Profiler.DARSHAN)
        if self.arg_parser.args.generate_data:
            self.data_generator = GeneratorFactory.get_generator(self.arg_parser.args.format)
        # Evaluation support
        self.do_eval = self.arg_parser.args.do_eval
        self.num_files_eval = self.arg_parser.args.num_files_eval
        self.batch_size_eval = self.arg_parser.args.batch_size_eval
        self.eval_time = self.arg_parser.args.eval_time
        self.eval_after_epoch = self.arg_parser.args.eval_after_epoch
        self.eval_every_epoch = self.arg_parser.args.eval_every_epoch
        

    def initialize(self):
        """
        Initializes the benchmark runtime.
        - It generates the required data
        - Start profiling session for Darshan and Tensorboard.
        """
        if self.arg_parser.args.debug and self.arg_parser.args.my_rank == 0:
            input("Press enter to start\n")
        self.framework.barrier()
        if self.arg_parser.args.generate_data:
            logging.info(f"{utcnow()} Starting data generation")
            self.data_generator.generate()
            logging.info(f"{utcnow()} Generation done")
        if self.arg_parser.args.profiling:
            self.darshan.start()
            self.framework.start_framework_profiler()
            self.framework.barrier()
            if self.arg_parser.args.my_rank == 0:
                logging.info(f"{utcnow()} Profiling Started")
        self.framework.barrier()

    def _eval(self, epoch_number):
        """
        Evaluation loop with a different sleep time than training.
        E.g. I believe in imseg, eval happens on CPU and time is pretty stable across runs
        """
        step = 1
        total = math.ceil(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
        for batch in self.framework.get_reader().next():
            if self.eval_time > 0:
                self.framework.compute(epoch_number, step, self.eval_time)
            step += 1
            if step > total:
                return step - 1
        return step - 1

    def _train(self, epoch_number):
        """
        Training loop for reading the dataset and performing training computations.
        :return: returns total steps.
        """
        step = 1
        total = math.ceil(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
        for batch in self.framework.get_reader().next():
            if self.computation_time > 0:
                self.framework.compute(epoch_number, step, self.computation_time)
            if self.arg_parser.args.checkpoint and step % self.arg_parser.args.steps_checkpoint == 0:
                self.framework.checkpoint(step)
            step += 1
            if step > total:
                return step - 1
            barrier()
        return step - 1

    def run(self):
        """
        Run the total epochs for training. 
        On each epoch, it prepares dataset for reading, it trains, and finalizes the dataset.
        If evaluation is enabled, it reads the eval dataset, performs evaluation and finalizes.
        """
        if not self.arg_parser.args.generate_only:
            # Print out the expected number of steps for each epoch and evaluation
            if self.my_rank == 0:
                total = math.ceil(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
                logging.info(f"{utcnow()} Steps per epoch: {total} = {self.num_samples} * {self.num_files_train} / {self.batch_size} / {self.comm_size} (samples per file * num files / batch size / comm size)")
                if self.do_eval:
                    total = math.ceil(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
                    logging.info(f"{utcnow()} Steps per eval: {total} = {self.num_samples} * {self.num_files_eval} / {self.batch_size_eval} / {self.comm_size} (samples per file * num files / batch size eval / comm size)")
            
            # Keep track of the next epoch at which we will evaluate
            next_eval_at = self.eval_after_epoch
            for epoch_number in range(1, self.arg_parser.args.epochs + 1):
                
                if self.my_rank == 0:
                    logging.info(f"{utcnow()} Starting epoch {epoch_number}")

                start_time = time()
                # Initialize the dataset
                self.framework.get_reader().read(epoch_number, do_eval=False)
                self.framework.barrier()

                if self.my_rank == 0:
                    logging.info(f"{utcnow()} Training dataset loaded for all ranks in {time() - start_time} seconds")
                
                start_time = time()
                steps = self._train(epoch_number)
                self.framework.barrier()

                if self.my_rank == 0:
                    logging.info(f"{utcnow()} Ending epoch {epoch_number} - {steps} steps completed in {time() - start_time} seconds")

                self.framework.get_reader().finalize()

                # Perform evaluation if enabled
                if self.do_eval and epoch_number == next_eval_at:
                    next_eval_at += self.eval_every_epoch
                
                    if self.my_rank == 0:
                        logging.info(f"{utcnow()} Starting eval")

                    start_time = time()
                    # Initialize the eval dataset
                    self.framework.get_reader().read(epoch_number, do_eval=True)
                    self.framework.barrier()

                    if self.my_rank == 0:
                        logging.info(f"{utcnow()} Eval dataset loaded for all ranks in {time() - start_time} seconds")
                    
                    start_time = time()
                    steps = self._eval(epoch_number)
                    self.framework.barrier()

                    if self.my_rank == 0:
                        logging.info(f"{utcnow()} Ending eval - {steps} steps completed in {time() - start_time} seconds")

                    self.framework.get_reader().finalize()

    def finalize(self):
        """
        It finalizes the dataset once training is completed.
        """
        self.framework.barrier()
        if not self.arg_parser.args.generate_only:
            if self.arg_parser.args.profiling:
                self.darshan.stop()
                self.framework.stop_framework_profiler.stop()
                self.framework.barrier()
                if self.my_rank == 0:
                    logging.info(f"{utcnow()} profiling stopped")
            if not self.arg_parser.args.keep_files:
                logging.info(f"{utcnow()} Keep files set to False. Deleting dataset")
                self.framework.barrier()
                if self.my_rank == 0:
                    if os.path.exists(self.arg_parser.args.data_folder):
                        shutil.rmtree(self.arg_parser.args.data_folder)
                        logging.info(f"{utcnow()} Deleted data files")
        self.framework.barrier()
        if self.my_rank == 0:
            logging.info(f"{utcnow()} Finalized for all ranks")


def main():
    """
        The main method to start the benchmark runtime.
        """
    os.environ["DARSHAN_DISABLE"] = "1"
    benchmark = DLIOBenchmark()
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    exit(0)
