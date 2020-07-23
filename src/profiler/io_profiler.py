from abc import ABC, abstractmethod

from src.utils.argument_parser import ArgumentParser


class IOProfiler(ABC):
    def __init__(self):
        self._arg_parser = ArgumentParser.get_instance()
        self.logdir = self._arg_parser.args.logdir

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass