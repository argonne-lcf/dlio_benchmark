from src.profiler.io_profiler import IOProfiler


class TFProfiler(IOProfiler):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if TFProfiler.__instance is None:
            TFProfiler()
        return TFProfiler.__instance

    def __init__(self):
        super().__init__()
        """ Virtually private constructor. """
        if TFProfiler.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            TFProfiler.__instance = self

    def start(self):
        pass

    def stop(self):
        pass