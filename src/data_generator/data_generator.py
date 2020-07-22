from abc import ABC, abstractmethod


class DataGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self):
        pass
