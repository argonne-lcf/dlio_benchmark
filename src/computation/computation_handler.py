from abc import ABC, abstractmethod


class ComputationHandler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute(self):
        pass
