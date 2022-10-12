from src.common.error_code import ErrorCodes
from src.framework.framework import Framework
from src.profiler.profiler_factory import ProfilerFactory
from src.common.enumerations import FrameworkType, Profiler, FormatType

import tensorflow as tf

from src.utils.argument_parser import ArgumentParser

print(tf.sysconfig.get_link_flags())
import horovod.tensorflow as hvd
import os

from src.reader.reader_factory import ReaderFactory

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
hvd.init()

class TFFramework(Framework):
    __instance = None

    def __init__(self, profiling):
        self.profiling = profiling
        if profiling:
            self.tensorboard = ProfilerFactory.get_profiler(Profiler.TENSORBOARD)
        self.reader_handler = None
        self.arg_parser = ArgumentParser.get_instance()

    def init_reader(self, format_type):
        if format_type == FormatType.DATA_LOADER:
            raise Exception(str(ErrorCodes.EC1001))
        self.reader_handler = ReaderFactory.get_format(format_type)

    def get_type(self):
        return FrameworkType.TENSORFLOW

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TFFramework.__instance is None:
            TFFramework.__instance = TFFramework(profiling)
        return TFFramework.__instance

    def barrier(self):
        """
        Barrier implementation using horovod's all-reduce
        """
        const = tf.constant(1)
        reduced = hvd.allreduce(const)

    def rank(self):
        return hvd.rank()

    def size(self):
        return hvd.size()

    def start_framework_profiler(self):
        if self.profiling:
            self.tensorboard.start()

    def stop_framework_profiler(self):
        if self.profiling:
            self.tensorboard.stop()

    def trace_object(self, string, step, r):
        return tf.profiler.experimental.Trace(string, step_num=step, _r=r)

    def compute(self, epoch_number, step, computation_time):
        tf.function(self.model)(epoch_number, step, computation_time)

    def get_reader(self):
        return self.reader_handler
