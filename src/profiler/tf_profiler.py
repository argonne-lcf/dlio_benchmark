from src.profiler.io_profiler import IOProfiler


class TFProfiler(IOProfiler):
    __instance = None

    @staticmethod
    def get_instance(logdir):
        """ Static access method. """
        if TFProfiler.__instance == None:
            TFProfiler(logdir)
        return TFProfiler.__instance

    def __init__(self,logdir):
        super().__init__(logdir)
        """ Virtually private constructor. """
        if TFProfiler.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            TFProfiler.__instance = self

    def start(self):
        pass

    def stop(self):
        pass