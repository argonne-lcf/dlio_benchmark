from src.common.error_code import ErrorCodes
from src.profiler.darshan_profiler import DarshanProfiler
from src.profiler.no_profiler import NoProfiler
from src.common.enumerations import Profiler
from src.profiler.tf_profiler import TFProfiler


class ProfilerFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_profiler(type):
        if type == Profiler.NONE:
            return NoProfiler()
        elif type == Profiler.DARSHAN:
            return DarshanProfiler.get_instance()
        elif type == Profiler.TENSORBOARD:
            return TFProfiler.get_instance()
        else:
            raise Exception(str(ErrorCodes.EC1001))