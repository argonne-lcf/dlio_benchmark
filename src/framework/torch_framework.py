from src.common.error_code import ErrorCodes
from src.common.enumerations import FormatType, FrameworkType
from src.framework.framework import Framework, DummyTraceObject

import torch
import functools
from time import sleep

import horovod.torch as hvd

from src.reader.reader_factory import ReaderFactory

hvd.init()

HANDLED_FUNCTIONS = {}

def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator

# Does this annotation mean that torch.mean will be replaced by torch_sleep?
@implements(torch.mean)
def torch_sleep(sleep_time):
    return sleep(sleep_time)

class TorchFramework(Framework):
    __instance = None

    def __init__(self, profiling):
        self.profiling = profiling
        self.reader_handler = None

    def init_reader(self, format_type):
        if format_type == FormatType.TFRECORD:
            raise Exception(str(ErrorCodes.EC1001))
        self.reader_handler = ReaderFactory.get_format(format_type)

    def get_type(self):
        return FrameworkType.PYTORCH

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TorchFramework.__instance is None:
            TorchFramework.__instance = TorchFramework(profiling)
        return TorchFramework.__instance

    def barrier(self):
        """
        Barrier implementation using horovod's all-reduce
        """
        const = torch.tensor(1)
        reduced = hvd.allreduce(const)

    def rank(self):
        return hvd.rank()

    def size(self):
        return hvd.size()

    def start_framework_profiler(self):
        pass

    def stop_framework_profiler(self):
        pass

    def trace_object(self, string, step, r):
        return DummyTraceObject(string, step, r)

    def compute(self, epoch_number, step, computation_time):
        torch_sleep(computation_time)

    def get_reader(self):
        return self.reader_handler
