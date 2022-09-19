"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of DLProfile
 DLIO is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
"""
from time import time

from src.common.enumerations import Profiler
from src.data_generator.generator_factory import GeneratorFactory
from src.framework.framework_factory import FrameworkFactory
from src.profiler.profiler_factory import ProfilerFactory
from src.utils.argument_parser import ArgumentParser
from src.utils.utility import utcnow

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import math
import shutil
import os
import logging


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
            format='%(message)s [%(pathname)s:%(lineno)d]'    # logging's max timestamp resolution is msecs, we will pass in microseconds in the message
        )

        self.framework = FrameworkFactory().get_framework(self.arg_parser.args.framework,
                                                          self.arg_parser.args.profiling)
        self.arg_parser.args.my_rank = self.framework.rank()
        self.arg_parser.args.comm_size = self.framework.size()
        self.framework.init_reader(self.arg_parser.args.format)
        self.darshan = None
        self.data_generator = None
        self.my_rank = self.arg_parser.args.my_rank
        self.comm_size = self.arg_parser.args.comm_size
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
            logging.info("{} Starting data generation".format(utcnow()))
            self.data_generator.generate()
            logging.info("{} Generation done".format(utcnow()))
        if self.arg_parser.args.profiling:
            self.darshan.start()
            self.framework.start_framework_profiler()
            self.framework.barrier()
            if self.arg_parser.args.my_rank == 0:
                print("profiling started")
        self.framework.barrier()

    def _eval(self, epoch_number):
        """
        Evaluation loop with a different sleep time than training.
        E.g. I believe in imseg, eval happens on CPU and time is pretty stable across runs
        """
        step = 1
        total = math.ceil(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
        for batch in self.framework.get_reader().next(do_eval=True):
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
        return step - 1

    def run(self):
        """
        Run the total epochs for training. On each epoch, it prepares dataset for reading, it trains, and finalizes the
        dataset.
        """
        if not self.arg_parser.args.generate_only:
            # Print out the expected number of steps for each epoch and evaluation
            if self.arg_parser.args.my_rank == 0:
                total = math.ceil(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
                logging.info("{} Steps per epoch: {} = {} * {} / {} / {} (samples per file * num files / batch size / comm size)".format(utcnow(), total, self.num_samples, self.num_files_train, self.batch_size, self.comm_size))
                if self.do_eval:
                    total = math.ceil(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
                    logging.info("{} Steps per eval: {} = {} * {} / {} / {} (samples per file * num files / batch size eval / comm size)".format(utcnow(), total, self.num_samples, self.num_files_eval, self.batch_size_eval, self.comm_size))
            
            next_eval_at = self.eval_after_epoch
            for epoch_number in range(1, self.arg_parser.args.epochs + 1):
                
                if self.arg_parser.args.my_rank == 0:
                    logging.info("{} Starting epoch {}".format(utcnow(), epoch_number))

                start_time = time()
                # Initialize the dataset
                self.framework.get_reader().read(epoch_number, do_eval=False)
                self.framework.barrier()
                if self.arg_parser.args.my_rank == 0:
                    logging.info("{} Training dataset loaded for all ranks in {} seconds".format(utcnow(), (time() - start_time)))
                
                start_time = time()
                steps = self._train(epoch_number)

                self.framework.barrier()
                if self.arg_parser.args.my_rank == 0:
                    logging.info("{} Ending epoch {} - {} steps completed in {} seconds".format(utcnow(), epoch_number, steps, time() - start_time))
                self.framework.get_reader().finalize()

                # Perform evaluation if enabled
                if self.do_eval and epoch_number == next_eval_at:
                    next_eval_at += self.eval_every_epoch
                
                    if self.arg_parser.args.my_rank == 0:
                        logging.info("{} Starting eval".format(utcnow()))

                    start_time = time()
                    # Initialize the eval dataset
                    self.framework.get_reader().read(epoch_number, do_eval=True)
                    self.framework.barrier()
                    if self.arg_parser.args.my_rank == 0:
                        logging.info("{} Eval dataset loaded for all ranks in {} seconds".format(utcnow(), (time() - start_time)))
                    
                    start_time = time()
                    steps = self._eval(epoch_number)
                    self.framework.barrier()
                    if self.arg_parser.args.my_rank == 0:
                        logging.info("{} Ending eval - {} steps completed in {} seconds".format(utcnow(), steps, time() - start_time))
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
                if self.arg_parser.args.my_rank == 0:
                    logging.info("{} profiling stopped".format(utcnow()))
            if not self.arg_parser.args.keep_files:
                logging.info("{} Keep files set to False. Deleting dataset", utcnow())
                self.framework.barrier()
                if self.arg_parser.args.my_rank == 0:
                    if os.path.exists(self.arg_parser.args.data_folder):
                        shutil.rmtree(self.arg_parser.args.data_folder)
                        logging.info("{} Deleted data files".format(utcnow()))
        self.framework.barrier()
        if self.arg_parser.args.my_rank == 0:
            logging.info("{} Finalized for all ranks".format(utcnow()))


if __name__ == '__main__':
    """
    The main method to start the benchmark runtime.
    """
    os.environ["DARSHAN_DISABLE"] = "1"
    benchmark = DLIOBenchmark()
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    exit(0)
