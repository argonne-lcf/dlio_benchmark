from abc import ABC, abstractmethod

from time import sleep
import horovod.tensorflow as hvd
import os

class DummyTraceObject(object):
    def __init__(self, string, step, r):
        pass

    def __enter__(self):
        return 1

    def __exit__(self, string, step, r):
        pass


class Framework(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def init_reader(self, format_type):
        pass

    @abstractmethod
    def barrier(self):
        pass

    def rank(self):
        return hvd.rank()

    def size(self):
        return hvd.size()

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
        """
        Performs Checkpointing for a specific step number. It writes different file of different sizes.
        """
        if not os.path.exists(self.arg_parser.args.output_folder):
            os.makedirs(self.arg_parser.args.output_folder)
        model_file = os.path.join(self.arg_parser.args.output_folder,
                                  "model_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        bak_file1 = os.path.join(self.arg_parser.args.output_folder,
                                 "file1_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        bak_file2 = os.path.join(self.arg_parser.args.output_folder,
                                 "file2_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        meta_file = os.path.join(self.arg_parser.args.output_folder,
                                 "meta_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        f = open(model_file, "w")
        # TODO: This could be parametrized as "model size" and affect the checkpoint writing time
        # E.g. BERT is a very big model, with 330M parameters, and would take longer to write.
        string_val = "x" * (1024 * 1024 * 4)
        f.write(string_val)
        f.close()
        f = open(bak_file1, "w")
        string_val = "x" * (1024 * 64)
        f.write(string_val)
        f.close()
        f = open(bak_file2, "w")
        string_val = "x" * (1024 * 4)
        f.write(string_val)
        f.close()
        f = open(meta_file, "w")
        string_val = "x" * (1024)
        f.write(string_val)
        f.close()
        pass

    def model(epoch, epoch_number, step, computation_time):
        sleep(computation_time)

    @abstractmethod
    def compute(self, epoch_number, step, computation_time):
        pass

    @abstractmethod
    def get_reader(self):
        pass
