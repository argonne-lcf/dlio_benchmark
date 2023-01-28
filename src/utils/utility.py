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
from src.utils.config import ConfigArguments
from time import time
from functools import wraps
import threading
import json

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_benchmark_test.log", mode = "a", encoding='utf-8'),
        logging.StreamHandler()
    ],format='[%(levelname)s] %(message)s'  # logging's max timestamp resolution is msecs, we will pass in usecs in the message
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
        return x, "%10.10f"%begin, "%10.10f"%end, os.getpid()
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

        if get_rank()==0:
            s = f'Resource usage information \n[PERFORMANCE] {"="*50}\n'
            s += f'[PERFORMANCE] Memory usage:\t\t {current / 10**6:.6f} MB \n'
            s += f'[PERFORMANCE] Peak memory usage:\t {peak / 10**6:.6f} MB \n'
            s += f'[PERFORMANCE] Time elapsed:\t\t {finish_time - start_time:.6f} s\n'
            s += f'[PERFORMANCE] {"="*50}\n'
            logging.info(s)
        tracemalloc.stop()
    return wrapper


def progress(count, total, status=''):
    """
    Printing a progress bar. Will be in the stdout when debug mode is turned on
    """
    _args = ConfigArguments.get_instance()
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + ">"+'-' * (bar_len - filled_len)
    if get_rank()==0:
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
