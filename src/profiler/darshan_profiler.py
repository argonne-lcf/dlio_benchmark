from src.profiler.io_profiler import IOProfiler


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
        pass

    def stop(self):
        pass