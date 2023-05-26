"""
   Copyright (c) 2022, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
from datetime import datetime
import logging
from time import time
from functools import wraps
import threading
import json
from typing import Dict

import numpy as np
import inspect

# UTC timestamp format with microsecond precision
from dlio_benchmark.common.enumerations import LoggerType

LOG_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
from mpi4py import MPI

def add_padding(n, num_digits=None):
    str_out = str(n)
    if num_digits!=None:
        return str_out.rjust(num_digits, "0")
    else:
        return str_out



def utcnow(format=LOG_TS_FORMAT):
    return datetime.now().strftime(format)


def get_rank():
    return MPI.COMM_WORLD.rank


def get_size():
    return MPI.COMM_WORLD.size


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = time()
        x = func(*args, **kwargs)
        end = time()
        return x, "%10.10f" % begin, "%10.10f" % end, os.getpid()

    return wrapper


import tracemalloc
from time import perf_counter


def measure_performance(func):
    '''Measure performance of a function'''

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = perf_counter()
        func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        finish_time = perf_counter()
        logging.basicConfig(format='%(asctime)s %(message)s')

        if get_rank() == 0:
            s = f'Resource usage information \n[PERFORMANCE] {"=" * 50}\n'
            s += f'[PERFORMANCE] Memory usage:\t\t {current / 10 ** 6:.6f} MB \n'
            s += f'[PERFORMANCE] Peak memory usage:\t {peak / 10 ** 6:.6f} MB \n'
            s += f'[PERFORMANCE] Time elapsed:\t\t {finish_time - start_time:.6f} s\n'
            s += f'[PERFORMANCE] {"=" * 50}\n'
            logging.info(s)
        tracemalloc.stop()

    return wrapper


def progress(count, total, status=''):
    """
    Printing a progress bar. Will be in the stdout when debug mode is turned on
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + ">" + '-' * (bar_len - filled_len)
    if get_rank() == 0:
        logging.info("\r[INFO] {} {}: [{}] {}% {} of {} ".format(utcnow(), status, bar, percents, count, total))
        if count == total:
            logging.info("")
        os.sys.stdout.flush()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def create_dur_event(name, cat, ts, dur, args={}):
    if "get_native_id" in dir(threading):
        tid = threading.get_native_id()
    elif "get_ident" in dir(threading):
        tid = threading.get_ident()
    else:
        tid = 0
    d = {
        "name": name,
        "cat": cat,
        "pid": get_rank(),
        "tid": tid,
        "ts": ts * 1000000,
        "dur": dur * 1000000,
        "ph": "X",
        "args": args
    }
    return d


class PerfTrace:
    __instance = None
    logger_type = LoggerType.DEFAULT
    logger = None
    def __init__(self):
        self.logfile = f"./.trace-{get_rank()}-of-{get_size()}" + ".pfw"
        self.log_file = None
        self.logger = None
        self.logger_type = LoggerType.DEFAULT
        PerfTrace.__instance = self

    @classmethod
    def get_instance(cls):
        """ Static access method. """
        if PerfTrace.__instance is None:
            PerfTrace()
        return PerfTrace.__instance

    @staticmethod
    def initialize_log(logdir, data_dir):
        instance = PerfTrace.get_instance()
        os.makedirs(logdir, exist_ok=True)
        instance.log_file = os.path.join(logdir, instance.logfile)
        if os.path.isfile(instance.log_file):
            os.remove(instance.log_file)
        if "DLIO_PROFILER" in os.environ and os.environ["DLIO_PROFILER"] == "DLIO_PROFILER":
            instance.logger_type = LoggerType.DLIO_PROFILER
            import dlio_profiler_py as dlio_logger
            instance.logger = dlio_logger
            instance.logger.initialize(instance.log_file, f"{data_dir}", process_id=get_rank())
        else:
            instance.logger = logging.getLogger("perftrace")
            instance.logger.setLevel(logging.DEBUG)
            instance.logger.propagate = False
            fh = logging.FileHandler(instance.log_file)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(message)s")
            fh.setFormatter(formatter)
            instance.logger.addHandler(fh)
            instance.logger.debug("[")

    def get_time(self):
        if self.logger_type == LoggerType.DLIO_PROFILER:
            return self.logger.get_time()
        else:
            return time()

    def log_event(self, name, cat, start_time, duration,  int_args=None):
        if int_args is None:
            int_args = {}
        if self.logger_type == LoggerType.DLIO_PROFILER:
            self.logger.log_event(name=name, cat=cat, start_time=start_time, duration=duration, int_args=int_args)
        else:
            event = create_dur_event(name, cat, start_time, duration, args=int_args)
            self.logger.debug(json.dumps(event, cls=NpEncoder))

    def finalize(self):
        if self.logger_type == LoggerType.DLIO_PROFILER:
            self.logger.finalize()
        else:
            self.logger.debug("]")


class Profile(object):

    def __init__(self, cat, name=None, epoch=None, step=None, image_idx=None, image_size=None):
        if not name:
            name = inspect.stack()[1].function
        self._name = name
        self._cat = cat
        self._arguments:Dict[str, int] = {}
        if epoch is not None: self._arguments["epoch"] = epoch
        if step is not None: self._arguments["step"] = step
        if image_idx is not None: self._arguments["image_idx"] = image_idx
        if image_size is not None: self._arguments["image_size"] = image_size
        self.reset()

    def __enter__(self):
        self._t1 = PerfTrace.get_instance().get_time()
        return self

    def update(self, epoch=None, step=None, image_idx=None, image_size=None):

        if epoch is not None: self._arguments["epoch"] = epoch
        if step is not None: self._arguments["step"] = step
        if image_idx is not None: self._arguments["image_idx"] = image_idx
        if image_size is not None: self._arguments["image_size"] = image_size
        return self

    def flush(self):
        self._t2 = PerfTrace.get_instance().get_time()
        if len(self._arguments) > 0:
            PerfTrace.get_instance().log_event(name=self._name, cat=self._cat, start_time=self._t1, duration=self._t2 - self._t1,
                                  int_args=self._arguments)
        else:
            PerfTrace.get_instance().log_event(name=self._name, cat=self._cat, start_time=self._t1, duration=self._t2 - self._t1)
        self._flush = True
        return self

    def reset(self):
        self._t1 = PerfTrace.get_instance().get_time()
        self._t2 = self._t1
        self._flush = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._flush:
            self.flush()

    def log(self, func):
        arg_names = inspect.getfullargspec(func)[0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            if "self" == arg_names[0]:
                if hasattr(args[0], "epoch"):
                    self._arguments["epoch"] = args[0].epoch
                if hasattr(args[0], "step"):
                    self._arguments["step"] = args[0].step
                if hasattr(args[0], "image_size"):
                    self._arguments["image_size"] = args[0].image_size
                if hasattr(args[0], "image_idx"):
                    self._arguments["image_idx"] = args[0].image_idx
            for name, value in zip(arg_names[1:], kwargs):
                if hasattr(args, name):
                    setattr(args, name, value)
                    if name == "epoch":
                        self._arguments["epoch"] = int(value)
                    elif name == "image_idx":
                        self._arguments["image_idx"] = int(value)
                    elif name == "image_size":
                        self._arguments["image_size"] = int(value)
                    elif name == "step":
                        self._arguments["image_size"] = int(value)

            start = PerfTrace.get_instance().get_time()
            x = func(*args, **kwargs)
            end = PerfTrace.get_instance().get_time()
            if len(self._arguments) > 0:
                PerfTrace.get_instance().log_event(name=func.__qualname__, cat=self._cat, start_time=start, duration=end - start,
                                      int_args=self._arguments)
            else:
                PerfTrace.get_instance().log_event(name=func.__qualname__, cat=self._cat, start_time=start, duration=end - start)
            return x

        return wrapper

    def iter(self, func, iter_name="step"):
        self._arguments[iter_name] = 1
        name = f"{inspect.stack()[1].function}.iter"
        kernal_name = f"{inspect.stack()[1].function}"
        start = PerfTrace.get_instance().get_time()
        for v in func:
            end = PerfTrace.get_instance().get_time()
            t0 = PerfTrace.get_instance().get_time()
            yield v
            t1 = PerfTrace.get_instance().get_time()

            if len(self._arguments) > 0:
                PerfTrace.get_instance().log_event(name=name, cat=self._cat, start_time=start, duration=end - start, int_args=self._arguments)
                PerfTrace.get_instance().log_event(name=kernal_name, cat=self._cat, start_time=t0, duration=t1 - t0, int_args=self._arguments)
            else:
                PerfTrace.get_instance().log_event(name=name, cat=self._cat, start_time=start, duration=end - start)
                PerfTrace.get_instance().log_event(name=kernal_name, cat=self._cat, start_time=t0, duration=t1 - t0)
            self._arguments[iter_name] += 1
            start = PerfTrace.get_instance().get_time()

    def log_init(self, init):
        arg_names = inspect.getfullargspec(init)[0]

        @wraps(init)
        def new_init(args, *kwargs):
            for name, value in zip(arg_names[1:], kwargs):
                setattr(args, name, value)
                if name == "epoch":
                    self._arguments["epoch"] = value
            start = PerfTrace.get_instance().get_time()
            init(args, *kwargs)
            end = PerfTrace.get_instance().get_time()

            if len(self._arguments) > 0:
                PerfTrace.get_instance().log_event(name=init.__qualname__, cat=self._cat, start_time=start, duration=end - start,
                                      int_args=self._arguments)
            else:
                PerfTrace.get_instance().log_event(name=init.__qualname__, cat=self._cat, start_time=start, duration=end - start)
        return new_init
