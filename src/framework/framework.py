from abc import ABC, abstractmethod

from time import sleep
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
        """
        Performs Checkpointing for a specific step number. It writes different file of different sizes.
        TODO: Parametrize the model size, which is vastly different e.g. between BERT, UNET3D and DLRM.
        TODO: Implement framework specific checkpointing - probably they write different number of files
        """
        output_folder = self.arg_parser.args.output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        model_file = os.path.join(output_folder, f"model_{step_number}_{self.arg_parser.args.my_rank}.bin")
        bak_file1 = os.path.join(output_folder, f"file1_{step_number}_{self.arg_parser.args.my_rank}.bin")
        bak_file2 = os.path.join(output_folder, f"file2_{step_number}_{self.arg_parser.args.my_rank}.bin")
        meta_file = os.path.join(output_folder, f"meta_{step_number}_{self.arg_parser.args.my_rank}.bin")

        f = open(model_file, "w")
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
