from src.profiler.io_profiler import IOProfiler


class DarshanProfiler(IOProfiler):
    __instance = None

    @staticmethod
    def get_instance(logdir):
        """ Static access method. """
        if DarshanProfiler.__instance == None:
            DarshanProfiler(logdir)
        return DarshanProfiler.__instance

    def __init__(self,logdir):
        super().__init__(logdir)
        """ Virtually private constructor. """
        if DarshanProfiler.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DarshanProfiler.__instance = self

    def start(self):
        pass

    def stop(self):
        pass