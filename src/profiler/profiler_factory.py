from src.common.error_code import ErrorCodes
from src.format.csv_format import CSVFormat
from src.format.hdf5_format import HDF5Format
from src.profiler.no_profiler import NoProfiler
from src.common.enumerations import Profiler


class ProfilerFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_profiler(type):
        if type == Profiler.NONE:
            return NoProfiler()
        elif type == Profiler.DARSHAN:
            return HDF5Format()
        elif type == Profiler.TENSORBOARD:
            return CSVFormat()
        else:
            raise Exception(str(ErrorCodes.EC1001))