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

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_benchmark_test.log", mode="a", encoding='utf-8'),
        logging.StreamHandler()
    ], format='[%(levelname)s] %(message)s'
    # logging's max timestamp resolution is msecs, we will pass in usecs in the message
)

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


def create_dur_event(name, cat, ts, dur):
    return {
        "name": name,
        "cat": cat,
        "pid": get_rank(),
        "tid": threading.get_native_id(),
        "ts": ts * 1000000,
        "dur": dur * 1000000,
        "ph": "X"
    }


def create_event(name, cat, ph):
    return {
        "name": name,
        "cat": cat,
        "pid": get_rank(),
        "tid": threading.get_native_id(),
        "ts": time() * 1000000,
        "ph": ph
    }


class PerfTrace:
    __instance = None

    def __init__(self):
        self.logfile = f"./.trace-{get_rank()}-of-{get_size()}" + ".pfw"
        self.log_file = None
        self.logger = None
        PerfTrace.__instance = self

    @classmethod
    def get_instance(cls):
        """ Static access method. """
        if PerfTrace.__instance is None:
            PerfTrace()
        return PerfTrace.__instance

    @staticmethod
    def initialize_log(logdir):
        instance = PerfTrace.get_instance()
        instance.log_file = os.path.join(logdir, instance.logfile)
        if os.path.isfile(instance.log_file):
            os.remove(instance.log_file)
        os.makedirs(logdir, exist_ok=True)
        instance.flush_log("")

    def event_complete(self, name, cat, ts, dur):
        event = create_dur_event(name, cat, ts, dur)
        self.flush_log(json.dumps(event))

    def event_start(self, name, cat='default'):
        event = create_event(name, cat, 'B')
        self.flush_log(json.dumps(event))

    def event_stop(self, name, cat='default'):
        event = create_event(name, cat, "E")
        self.flush_log(json.dumps(event))

    def flush_log(self, s):
        if self.logger is None:
            if os.path.isfile(self.log_file):
                started = True
            else:
                started = False
            self.logger = logging.getLogger("perftrace")
            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False
            fh = logging.FileHandler(self.log_file)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.debug("[")
        if s != "":
            self.logger.debug(f"{s},")

    def finalize(self):
        start = time()
        end = time()
        event = create_dur_event("finalize", "dlio_benchmark", start, dur=end - start)
        self.logger.debug(f"{json.dumps(event)}]")


def event_logging(module):
    def event_logging_f(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            x = func(*args, **kwargs)
            end = time()
            instance = PerfTrace.get_instance()
            event = create_dur_event(func.__qualname__, module, start, dur=end - start)
            instance.flush_log(json.dumps(event))
            return x

        return wrapper

    return event_logging_f
