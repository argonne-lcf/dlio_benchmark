from abc import ABC, abstractmethod


class IOProfiler(ABC):
    def __init__(self, log_dir):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass