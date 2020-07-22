from abc import ABC, abstractmethod


class FormatHandler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def write(self):
        pass

    @abstractmethod
    def read(self):
        pass
