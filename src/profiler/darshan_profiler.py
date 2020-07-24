from src.profiler.io_profiler import IOProfiler
import os

class DarshanProfiler(IOProfiler):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if DarshanProfiler.__instance is None:
            DarshanProfiler()
        return DarshanProfiler.__instance

    def __init__(self):
        super().__init__()
        """ Virtually private constructor. """
        if DarshanProfiler.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DarshanProfiler.__instance = self

    def start(self):
        os.environ["LD_PRELOAD"] = "=/soft/perftools/darshan/darshan-3.1.8/lib/libdarshan.so"
        os.environ["DXT_ENABLE_IO_TRACE"] = "1"
        os.environ["DARSHAN_DISABLE"] = "0"

    def stop(self):
        del os.environ['LD_PRELOAD']
        del os.environ['DXT_ENABLE_IO_TRACE']
