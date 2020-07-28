from src.common.enumerations import Profiler
from src.data_generator.generator_factory import GeneratorFactory
from src.reader.reader_factory import ReaderFactory
from src.profiler.profiler_factory import ProfilerFactory
from src.utils.argument_parser import ArgumentParser
import tensorflow as tf
import horovod.tensorflow as hvd
import math
import os
import shutil

hvd.init()
from mpi4py import MPI


def barrier():
    const = tf.constant(1)
    reduced = hvd.allreduce(const)


class DLIOBenchmark(object):
    def __init__(self):
        self.arg_parser = ArgumentParser.get_instance()
        self.darshan = None
        self.tensorboard = None
        self.data_generator = None
        self.my_rank = self.arg_parser.args.my_rank
        self.comm_size = self.arg_parser.args.comm_size
        self.num_files = self.arg_parser.args.num_files
        self.num_samples = self.arg_parser.args.num_samples
        self.batch_size = self.arg_parser.args.batch_size
        if self.arg_parser.args.profiling:
            self.darshan = ProfilerFactory().get_profiler(Profiler.DARSHAN)
            self.tensorboard = ProfilerFactory().get_profiler(Profiler.TENSORBOARD)
        if self.arg_parser.args.generate_data:
            self.data_generator = GeneratorFactory.get_generator(self.arg_parser.args.format)
        self.reader_handler = ReaderFactory.get_format(self.arg_parser.args.format)

    def initialize(self):
        if self.arg_parser.args.debug and self.arg_parser.args.my_rank == 0:
            input("Press enter to start\n")
        barrier()
        if self.arg_parser.args.generate_data:
            self.data_generator.generate()
            print("Generation done")
        if self.arg_parser.args.profiling:
            self.darshan.start()
            self.tensorboard.start()
            print("profiling started")
        barrier()

    def _checkpoint(self, step_number):
        if not os.path.exists(self.arg_parser.args.output_folder):
            os.makedirs(self.arg_parser.args.output_folder)
        model_file = os.path.join(self.arg_parser.args.output_folder,
                                  "model_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        bak_file1 = os.path.join(self.arg_parser.args.output_folder,
                                 "file1_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        bak_file2 = os.path.join(self.arg_parser.args.output_folder,
                                 "file2_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        meta_file = os.path.join(self.arg_parser.args.output_folder,
                                 "meta_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        f = open(model_file, "w")
        string_val = "x" * (1024 * 1024 * 4)
        f.write(string_val)
        f.close()
        f = open(bak_file1, "w")
        string_val = "x" * (1024 * 64)
        f.write(string_val)
        f.close()
        f = open(bak_file2, "w")
        string_val = "x" * (1024 * 4)
        f.write(string_val)
        f.close()
        f = open(meta_file, "w")
        string_val = "x" * (1024)
        f.write(string_val)
        f.close()
        pass

    def _train(self):
        step = 1

        total = math.ceil(self.num_samples * self.num_files / self.batch_size / self.comm_size)
        for element in self.reader_handler.next():
            if self.arg_parser.args.checkpoint and step % self.arg_parser.args.steps_checkpoint == 0:
                self._checkpoint(step)
            step += 1
            if step > total:
                return step - 1
        return step - 1

    def run(self):
        if not self.arg_parser.args.generate_only:
            for epoch_number in range(0, self.arg_parser.args.epochs):
                self.reader_handler.read(epoch_number)
                print("Datasets loaded in {} epochs for rank {}".format(epoch_number + 1, self.arg_parser.args.my_rank))
                steps = self._train()
                print("Finished {} steps in {} epochs for rank {}".format(steps, epoch_number + 1,
                                                                          self.arg_parser.args.my_rank))
                self.reader_handler.finalize()

    def finalize(self):
        print("Finalizing for rank {}".format(self.arg_parser.args.my_rank))
        barrier()
        if not self.arg_parser.args.generate_only:
            if self.arg_parser.args.profiling:
                self.darshan.stop()
                self.tensorboard.stop()
                print("profiling stopped")
            if not self.arg_parser.args.keep_files:
                barrier()
                if self.arg_parser.args.my_rank == 0:
                    if os.path.exists(self.arg_parser.args.data_folder):
                        shutil.rmtree(self.arg_parser.args.data_folder)
                        print("Deleted data files")
        #


if __name__ == '__main__':
    os.environ["DARSHAN_DISABLE"] = "1"
    benchmark = DLIOBenchmark()
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    barrier()
    exit(0)
