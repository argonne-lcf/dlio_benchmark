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
import numpy as np
import inspect
import dlio_profiler_py as dlio_logger

# UTC timestamp format with microsecond precision
LOG_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
from mpi4py import MPI


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
    d = {
        "name": name,
        "cat": cat,
        "pid": get_rank(),
        "tid": threading.get_native_id(),
        "ts": ts * 1000000,
        "dur": dur * 1000000,
        "ph": "X",
        "args": args
    }
    return d


class PerfTrace:
    __instance = None

    def __init__(self):
        self.logfile = f"./.trace-{get_rank()}-of-{get_size()}" + ".pfw"
        self.log_file = None
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
        dlio_logger.initialize(instance.log_file, data_dir)

    def finalize(self):
        dlio_logger.finalize()


class Profile(object):

    def __init__(self, cat, name=None, epoch=None, step=None, image_idx=None, image_size=None):
        if not name:
            name = inspect.stack()[1].function
        dlio_logger.reset()
        self._name = name
        self._cat = cat
        if epoch is not None: dlio_logger.update_int("epoch", epoch)
        if step is not None: dlio_logger.update_int("step", step)
        if image_idx is not None: dlio_logger.update_int("image_idx", image_idx)
        if image_size is not None: dlio_logger.update_int("image_size", image_size)
        self.reset()

    def __enter__(self):
        dlio_logger.start(self._name, self._cat)
        return self

    def update(self, epoch=None, step=None, image_idx=None, image_size=None):

        if epoch is not None: dlio_logger.update_int("epoch", epoch)
        if step is not None: dlio_logger.update_int("step", step)
        if image_idx is not None: dlio_logger.update_int("image_idx", image_idx)
        if image_size is not None: dlio_logger.update_int("image_size", image_size)
        return self

    def reset(self):
        dlio_logger.start(self._name, self._cat)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dlio_logger.stop()

    def log(self, func):
        arg_names = inspect.getfullargspec(func)[0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            if "self" == arg_names[0]:
                if hasattr(args[0], "epoch"):
                    dlio_logger.update_int("epoch", args[0].epoch)
                if hasattr(args[0], "step"):
                    dlio_logger.update_int("step", args[0].step)
                if hasattr(args[0], "image_size"):
                    dlio_logger.update_int("image_size", args[0].image_size)
                if hasattr(args[0], "image_idx"):
                    dlio_logger.update_int("image_idx", args[0].image_idx)
            for name, value in zip(arg_names[1:], kwargs):
                if hasattr(args, name):
                    setattr(args, name, value)
                    if name == "epoch":
                        dlio_logger.update_int("epoch", value)
                    elif name == "image_idx":
                        dlio_logger.update_int("image_idx", value)
                    elif name == "image_size":
                        dlio_logger.update_int("image_size", value)
                    elif name == "step":
                        dlio_logger.update_int("step", value)
            dlio_logger.start(func.__qualname__, self._cat)
            x = func(*args, **kwargs)
            dlio_logger.stop()
            return x

        return wrapper

    def iter(self, func, iter_name="step"):
        iter_value = 1
        dlio_logger.update_int(iter_name, iter_value)
        name = f"{inspect.stack()[1].function}.iter"
        kernal_name = f"{inspect.stack()[1].function}"
        dlio_logger.start(name, self._cat)
        for v in func:
            dlio_logger.stop()
            dlio_logger.start(kernal_name, self._cat)
            yield v
            dlio_logger.stop()
            iter_value += 1
            dlio_logger.update_int(iter_name, iter_value)

    def log_init(self, init):
        arg_names = inspect.getfullargspec(init)[0]

        @wraps(init)
        def new_init(args, *kwargs):
            for name, value in zip(arg_names[1:], kwargs):
                setattr(args, name, value)
                if name == "epoch":
                    dlio_logger.update_int("epoch", value)
            dlio_logger.start(init.__qualname__, self._cat)
            init(args, *kwargs)
            dlio_logger.stop()

        return new_init
