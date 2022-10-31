from abc import ABC, abstractmethod
from src.utils.utility import utcnow

from time import sleep
import os
import logging

from utils.argument_parser import ArgumentParser

class DummyTraceObject(object):
    def __init__(self, string, step, r):
        pass

    def __enter__(self):
        return 1

    def __exit__(self, string, step, r):
        pass


class Framework(ABC):
    def __init__(self):
        self.args = ArgumentParser.get_instance().args
        self.output_folder = self.args.output_folder
        pass

    @abstractmethod
    def init_reader(self, format_type):
        pass

    @abstractmethod 
    def get_type(self):
        pass
    
    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def rank(self):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def start_framework_profiler(self):
        pass

    @abstractmethod
    def stop_framework_profiler(self):
        pass

    @abstractmethod
    def trace_object(self, string, step, r):
        pass

    def checkpoint(self, step_number):
        pass

    def model(epoch, epoch_number, step, computation_time):
        sleep(computation_time)

    @abstractmethod
    def compute(self, epoch_number, step, computation_time):
        pass

    @abstractmethod
    def get_reader(self):
        pass
