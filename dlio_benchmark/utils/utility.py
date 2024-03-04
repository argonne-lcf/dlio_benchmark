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
import pathlib
from enum import Enum

import numpy as np
import inspect
import psutil
import socket
import importlib.util
# UTC timestamp format with microsecond precision
from dlio_benchmark.common.enumerations import LoggerType, MPIState

LOG_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

# MPI cannot be initialized automatically, or read_thread spawn/forkserver
# child processes will abort trying to open a non-existant PMI_fd file.
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

p = psutil.Process()


def add_padding(n, num_digits=None):
    str_out = str(n)
    if num_digits != None:
        return str_out.rjust(num_digits, "0")
    else:
        return str_out


def utcnow(format=LOG_TS_FORMAT):
    return datetime.now().strftime(format)


# After the DLIOMPI singleton has been instantiated, the next call must be
# either initialize() if in an MPI process, or set_parent_values() if in a
# non-MPI pytorch read_threads child process.
class DLIOMPI:
    __instance = None

    def __init__(self):
        if DLIOMPI.__instance is not None:
            raise Exception(f"Class {self.classname()} is a singleton!")
        else:
            self.mpi_state = MPIState.UNINITIALIZED
            DLIOMPI.__instance = self

    @staticmethod
    def get_instance():
        if DLIOMPI.__instance is None:
            DLIOMPI()
        return DLIOMPI.__instance

    @staticmethod
    def reset():
        DLIOMPI.__instance = None

    @classmethod
    def classname(cls):
        return cls.__qualname__

    def initialize(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            # MPI may have already been initialized by dlio_benchmark_test.py
            if not MPI.Is_initialized():
                MPI.Init()
            self.mpi_state = MPIState.MPI_INITIALIZED
            self.mpi_rank = MPI.COMM_WORLD.rank
            self.mpi_size = MPI.COMM_WORLD.size
            self.mpi_world = MPI.COMM_WORLD
        elif self.mpi_state == MPIState.CHILD_INITIALIZED:
            raise Exception(f"method {self.classname()}.initialize() called in a child process")
        else:
            pass    # redundant call

    # read_thread processes need to know their parent process's rank and comm_size,
    # but are not MPI processes themselves.
    def set_parent_values(self, parent_rank, parent_comm_size):
        if self.mpi_state == MPIState.UNINITIALIZED:
            self.mpi_state = MPIState.CHILD_INITIALIZED
            self.mpi_rank = parent_rank
            self.mpi_size = parent_comm_size
            self.mpi_world = None
        elif self.mpi_state == MPIState.MPI_INITIALIZED:
            raise Exception(f"method {self.classname()}.set_parent_values() called in a MPI process")
        else:
            raise Exception(f"method {self.classname()}.set_parent_values() called twice")

    def rank(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.rank() called before initializing MPI")
        else:
            return self.mpi_rank

    def size(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.size() called before initializing MPI")
        else:
            return self.mpi_size

    def comm(self):
        if self.mpi_state == MPIState.MPI_INITIALIZED:
            return self.mpi_world
        elif self.mpi_state == MPIState.CHILD_INITIALIZED:
            raise Exception(f"method {self.classname()}.comm() called in a child process")
        else:
            raise Exception(f"method {self.classname()}.comm() called before initializing MPI")

    def finalize(self):
        if self.mpi_state == MPIState.MPI_INITIALIZED and MPI.Is_initialized():
            MPI.Finalize()

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

        if DLIOMPI.get_instance().rank() == 0:
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
    if DLIOMPI.get_instance().rank() == 0:
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
    args["hostname"] = socket.gethostname()
    args["cpu_affinity"] = p.cpu_affinity()
    d = {
        "name": name,
        "cat": cat,
        "pid": DLIOMPI.get_instance().rank(),
        "tid": tid,
        "ts": ts * 1000000,
        "dur": dur * 1000000,
        "ph": "X",
        "args": args
    }
    return d

  
def get_trace_name(output_folder, use_pid=False):
    val = ""
    if use_pid:
        val = f"-{os.getpid()}"
    return f"{output_folder}/trace-{DLIOMPI.get_instance().rank()}-of-{DLIOMPI.get_instance().size()}{val}.pfw"
