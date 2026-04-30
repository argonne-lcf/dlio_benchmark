"""
   Copyright (c) 2025, UChicago Argonne, LLC
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
from time import time, sleep as base_sleep
from functools import wraps
import threading
import json
import socket
import argparse

import psutil
import numpy as np

# Try to import dgen-py for high-performance data generation (30-50x faster than NumPy)
try:
    import dgen_py
    HAS_DGEN = True
except ImportError:
    HAS_DGEN = False
    dgen_py = None

# ── Process-level singleton dgen_py.Generator ─────────────────────────────────
#
# Creating a dgen_py.Generator spawns a private Rayon thread pool (N OS threads).
# Destroying and recreating it for every file costs ~10–50 ms per file and becomes
# catastrophic for workloads with many small objects (JPEG/PNG < 1 MB: pool creation
# dominates actual data generation).
#
# Design:
#   ONE Generator per MPI process (each MPI worker IS a separate OS process).
#   Reused across ALL gen_random_tensor() calls via reset() + set_seed() + get_chunk().
#
#   Small objects (< _DGEN_SMALL_THRESHOLD = 1 MiB): skip the Generator entirely
#   and call dgen_py.generate_buffer() which uses a thread-local RollingPool —
#   no Rayon thread pool at all, ~1.7 GB/s for 64 KB objects.
#
#   Large objects (>= 1 MiB): use the singleton Generator.  Thread pool is created
#   ONCE and reused; reset()+set_seed() between files is O(µs).
#
# Thread safety: generation always happens in the main thread (the async pipeline
# submits uploads to a pool but generates sequentially in the main thread), so the
# singleton is only accessed from one thread at a time.

_DGEN_PROC_GEN = None          # type: Optional['dgen_py.Generator']
_DGEN_PROC_GEN_CAPACITY = 0    # bytes: current singleton's total_size
_DGEN_PROC_GEN_LOCK = threading.Lock()  # One-time lazy-init guard
_DGEN_SMALL_THRESHOLD = 1 * 1024 * 1024  # 1 MiB — route smaller objects to rolling pool


def _get_dgen_proc_gen(total_bytes: int) -> 'dgen_py.Generator':
    """Return (or lazily create) the process-level singleton dgen_py.Generator.

    Recreated only when *total_bytes* exceeds the current capacity — this happens
    at most once per process lifetime (the first large-object request sets the
    capacity; subsequent requests of any smaller size reuse the same instance).
    """
    global _DGEN_PROC_GEN, _DGEN_PROC_GEN_CAPACITY
    # Fast path — already created and large enough.
    if _DGEN_PROC_GEN is not None and total_bytes <= _DGEN_PROC_GEN_CAPACITY:
        return _DGEN_PROC_GEN
    with _DGEN_PROC_GEN_LOCK:
        # Re-check under lock (another thread may have initialised between the
        # fast-path check and acquiring the lock).
        if _DGEN_PROC_GEN is None or total_bytes > _DGEN_PROC_GEN_CAPACITY:
            # Minimum 256 MiB so small fluctuations in file size don't trigger
            # repeated recreation.  Larger files expand the capacity once.
            new_capacity = max(total_bytes, 256 * 1024 * 1024)
            _DGEN_PROC_GEN = dgen_py.Generator(size=new_capacity)
            _DGEN_PROC_GEN_CAPACITY = new_capacity
    return _DGEN_PROC_GEN

from dlio_benchmark.common.enumerations import MPIState

# Try to load dftracer. If DFTRACER_ENABLE=1 is set and the library is installed,
# use the real dftracer decorators so .pfw trace files are written.
# Fall back to no-op stubs when the library is absent or DFTRACER_ENABLE is not set.
try:
    from dftracer.python import (
        dftracer as PerfTrace,
        dft_fn as Profile,
        ai as dft_ai,
        DFTRACER_ENABLE,
    )
except Exception:
    DFTRACER_ENABLE = False

    class _NoOpFn:
        """No-op stub for dft_fn (Profile context manager / decorator)."""
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getattr__(self, name): return _NoOpFn()
        def __call__(self, fn=None, *args, **kwargs):
            if callable(fn):
                return fn
            if fn is not None:
                return fn   # pass iterables through (e.g. dft_ai.x.iter(iterable))
            return self
        def log(self, fn=None, *args, **kwargs):
            if callable(fn): return fn
            return lambda f: f
        def log_init(self, fn=None, *args, **kwargs):
            if callable(fn): return fn
            return lambda f: f
        def update(self, *args, **kwargs): pass

    class _NoOpTracer:
        """No-op stub for dftracer singleton."""
        @staticmethod
        def get_instance(): return _NoOpTracer()
        def initialize(self, *a, **kw): pass
        def finalize(self, *a, **kw): pass
        def get_time(self): return 0
        def enter_event(self): pass
        def exit_event(self): pass
        def log_event(self, *a, **kw): pass
        def log_metadata_event(self, *a, **kw): pass

    class _NoOpAI:
        """No-op stub for dft_ai — supports @dft_ai, @dft_ai.x.y, dft_ai.x.iter(it)."""
        def __call__(self, fn=None, *args, **kwargs):
            if callable(fn): return fn
            if fn is not None: return fn
            return self
        def __getattr__(self, name): return _NoOpFn()
        def update(self, *args, **kwargs): pass

    Profile = _NoOpFn
    PerfTrace = _NoOpTracer
    dft_ai = _NoOpAI()

LOG_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

OUTPUT_LEVEL = 35
logging.addLevelName(OUTPUT_LEVEL, "OUTPUT")
def output(self, message, *args, **kwargs):
    if self.isEnabledFor(OUTPUT_LEVEL):
        self._log(OUTPUT_LEVEL, message, args, **kwargs)
logging.Logger.output = output

class DLIOLogger:
    __instance = None

    def __init__(self):
        self.logger = logging.getLogger("DLIO")
        #self.logger.setLevel(logging.DEBUG)
        if DLIOLogger.__instance is not None:
            raise Exception(f"Class {self.classname()} is a singleton!")
        else:
            DLIOLogger.__instance = self
    @staticmethod
    def get_instance():
        if DLIOLogger.__instance is None:
            DLIOLogger()
        return DLIOLogger.__instance.logger
    @staticmethod
    def reset():
        DLIOLogger.__instance = None
# MPI cannot be initialized automatically, or read_thread spawn/forkserver
# child processes will abort trying to open a non-existant PMI_fd file.
import mpi4py
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
        from mpi4py import MPI
        if self.mpi_state == MPIState.UNINITIALIZED:
            # MPI may have already been initialized by dlio_benchmark_test.py
            if not MPI.Is_initialized():
                MPI.Init()
            
            self.mpi_state = MPIState.MPI_INITIALIZED
            split_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
            # Number of processes on this node and local rank
            local_ppn = split_comm.size
            self.mpi_local_rank = split_comm.rank
            # Create a communicator of one leader per node
            if split_comm.rank == 0:
                leader_comm = MPI.COMM_WORLD.Split(color=0, key=MPI.COMM_WORLD.rank)
                # Gather each node's process count
                ppn_list = leader_comm.allgather(local_ppn)
            else:
                # Non-leaders do not participate
                MPI.COMM_WORLD.Split(color=MPI.UNDEFINED, key=MPI.COMM_WORLD.rank)
                ppn_list = None
            # Broadcast the per-node list to all processes
            self.mpi_ppn_list = MPI.COMM_WORLD.bcast(ppn_list, root=0)
            # Total number of nodes
            self.mpi_nodes = len(self.mpi_ppn_list)
            # Total world size and rank
            self.mpi_size = MPI.COMM_WORLD.size
            self.mpi_rank = MPI.COMM_WORLD.rank
            self.mpi_world = MPI.COMM_WORLD
            # Compute node index and per-node offset
            offsets = [0] + list(np.cumsum(self.mpi_ppn_list)[:-1])
            # Determine which node this rank belongs to
            for idx, off in enumerate(offsets):
                if self.mpi_rank >= off and self.mpi_rank < off + self.mpi_ppn_list[idx]:
                    self.mpi_node = idx
                    break
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

    def local_rank(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.size() called before initializing MPI")
        else:
            return self.mpi_local_rank

    def npernode(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.size() called before initializing MPI")
        else:
            return self.mpi_ppn_list[self.mpi_node]

    def ranks_per_node(self) -> int:
        """Number of MPI ranks sharing this physical node.

        Equivalent to npernode() in MPI_INITIALIZED state, but safe to call
        in CHILD_INITIALIZED state (where full topology is unavailable) —
        falls back to total comm_size as a conservative estimate.
        """
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.ranks_per_node() called before initializing MPI")
        elif self.mpi_state == MPIState.CHILD_INITIALIZED:
            # Child processes don't run through initialize(), so mpi_ppn_list
            # is not set.  Return comm_size as a conservative fallback so that
            # auto-sizing formulas (cpu_count // ranks_per_node) don't over-allocate.
            return self.mpi_size
        else:
            return self.mpi_ppn_list[self.mpi_node]

    def nnodes(self):
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.size() called before initializing MPI")
        else:
            return self.mpi_nodes
    
    def node(self):
        """
        Return the node index for this rank.
        """
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.node() called before initializing MPI")
        else:
            return self.mpi_node
    
    def reduce(self, num):
        from mpi4py import MPI
        if self.mpi_state == MPIState.UNINITIALIZED:
            raise Exception(f"method {self.classname()}.reduce() called before initializing MPI")
        else:
            return MPI.COMM_WORLD.allreduce(num, op=MPI.SUM)
    
    def finalize(self):
        from mpi4py import MPI
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


# Module-level state for the Rich progress bar used by progress()
_rich_progress_instance = None
_rich_progress_task_id = None


def progress(count, total, status=''):
    """Display a progress bar for data generation operations.

    Uses Rich when available (provides a proper animated spinner/bar), otherwise
    falls back to plain stdout writing.  The ``\\r``-in-logger approach used
    previously was unreliable in non-interactive terminals and log files.
    """
    global _rich_progress_instance, _rich_progress_task_id

    if DLIOMPI.get_instance().rank() != 0:
        return

    try:
        from rich.progress import (
            BarColumn, Progress, SpinnerColumn,
            TextColumn, TimeElapsedColumn,
        )

        # Create a fresh progress bar at the start of a new sequence
        if _rich_progress_instance is None or count == 1:
            if _rich_progress_instance is not None:
                try:
                    _rich_progress_instance.stop()
                except Exception:
                    pass
            _rich_progress_instance = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                transient=True,
            )
            _rich_progress_instance.start()
            _rich_progress_task_id = _rich_progress_instance.add_task(
                status, total=total
            )

        _rich_progress_instance.update(
            _rich_progress_task_id, completed=count, description=status
        )

        if count >= total:
            _rich_progress_instance.stop()
            _rich_progress_instance = None
            _rich_progress_task_id = None

    except Exception:
        # Fallback: write directly to stdout (no \r in log messages)
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + ">" + '-' * (bar_len - filled_len - 1)
        end = '\n' if count >= total else ''
        os.sys.stdout.write(
            f"\r[{bar}] {percents:.1f}%  {count}/{total}  {status}{end}"
        )
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
        
def sleep(config):
    sleep_time = 0.0
    if isinstance(config, dict) and len(config) > 0:
        if "type" in config:
            if config["type"] == "normal":
                sleep_time = np.random.normal(config["mean"], config["stdev"])
            elif config["type"] == "uniform":
                sleep_time = np.random.uniform(config["min"], config["max"])
            elif config["type"] == "gamma":
                sleep_time = np.random.gamma(config["shape"], config["scale"])
            elif config["type"] == "exponential":
                sleep_time = np.random.exponential(config["scale"])
            elif config["type"] == "poisson":
                sleep_time = np.random.poisson(config["lam"])
        else:
            if "mean" in config:
                if "stdev" in config:
                    sleep_time = np.random.normal(config["mean"], config["stdev"])
                else:
                    sleep_time = config["mean"]
    elif isinstance(config, (int, float)):
        sleep_time = config
    sleep_time = abs(sleep_time)
    if sleep_time > 0.0:
        base_sleep(sleep_time)
    return sleep_time

def gen_random_tensor(shape, dtype, rng=None, method=None, writeable=True, seed=None):
    """Generate random tensor data for DLIO benchmarks.

    DEFAULT: dgen-py (high-performance Rust-backed random data, zero-copy BytesView).
    This is 155x faster than NumPy and uses no extra memory during generation.

    The only supported methods are:
    - 'dgen'  : dgen-py (default, Python>=3.10). Falls back to numpy with a
                warning if dgen-py is not installed.
    - 'numpy' : NumPy random generation. Slow legacy path — only use for explicit
                comparison benchmarks. Set DLIO_DATA_GEN=numpy to activate.

    'auto' is intentionally NOT a supported default: silent fallback to numpy is
    a footgun — callers would get 155x slower generation without any indication.

    Args:
        shape:     Tuple specifying tensor dimensions.
        dtype:     NumPy dtype for the output array.
        rng:       Optional NumPy Generator (only used for the numpy slow path when
                   seed is not provided).
        method:    Explicit method override ('dgen' or 'numpy'). If None, reads
                   DLIO_DATA_GEN from the environment (default: 'dgen').
        writeable: If False, skip the extra .copy() in the dgen path, saving one
                   full array allocation. Safe when the caller only reads the array
                   (e.g. np.savez). npz_generator passes writeable=False.
        seed:      Optional integer seed for reproducible generation. When provided:
                   - dgen path: creates a dedicated dgen_py.Generator(seed=seed) so
                     the seed is passed all the way into the Rust RNG layer.  The
                     process singleton and generate_buffer() paths are bypassed when
                     a seed is supplied because they have no per-call seed support.
                   - numpy path: creates a new default_rng(seed=seed), ignoring rng
                   When None (default): uses the fast singleton/pool paths for maximum
                   throughput — entropy, not reproducibility, is the goal.
                   For MPI workloads, pass seed = BASE_SEED + file_index so each file
                   gets unique-but-reproducible data across runs.
    """
    # ── Method selection ────────────────────────────────────────────────────────
    # Default is 'dgen'. The environment can override to 'numpy' for explicit
    # comparison runs, but there is NO silent auto-fallback. If dgen-py is not
    # installed and 'dgen' is requested, we raise immediately rather than
    # silently producing correct-but-vastly-slower results.
    if method is None:
        method = os.environ.get('DLIO_DATA_GEN', 'dgen').lower()

    method = method.lower()

    use_dgen = (method == 'dgen')

    if method == 'numpy':
        # Explicit numpy request — allowed for comparison benchmarks only.
        use_dgen = False
    elif use_dgen and not HAS_DGEN:
        # dgen-py not installed (e.g. Python 3.9 where dgen-py is unavailable).
        # Warn once and fall back to numpy so the benchmark still runs.
        logging.getLogger("DLIO").warning(
            "dgen-py is not installed — falling back to NumPy for data generation "
            "(~155x slower). Install dgen-py>=0.2.0 (requires Python>=3.10) for "
            "full performance, or set DLIO_DATA_GEN=numpy to suppress this warning."
        )
        use_dgen = False
    
    # Fast path: Use dgen-py with ZERO-COPY BytesView (155x faster than NumPy)
    if use_dgen:
        total_size = int(np.prod(shape))
        element_size = np.dtype(dtype).itemsize
        total_bytes = total_size * element_size

        if seed is not None:
            # ── Seeded path: create a dedicated Generator so the seed takes effect ──
            # The singleton and generate_buffer() don't support per-call seeds.
            # Creating a Generator here spawns a Rayon thread pool, so this path
            # is intentionally only used when the caller explicitly requests a seed.
            gen = dgen_py.Generator(size=max(total_bytes, 256 * 1024 * 1024), seed=seed)
            bytesview = gen.get_chunk(total_bytes)
        elif total_bytes < _DGEN_SMALL_THRESHOLD:
            # ── Small objects (< 1 MiB): thread-local RollingPool ────────────
            # generate_buffer() uses a per-OS-thread RollingPool that generates
            # one BLOCK_SIZE (1 MiB) backing buffer once and hands out zero-copy
            # Arc-counted slices.  No Rayon thread pool is created — overhead
            # is O(µs), ideal for JPEG/PNG workloads generating millions of
            # small objects.
            bytesview = dgen_py.generate_buffer(total_bytes)
        else:
            # ── Large objects (>= 1 MiB): process-level singleton Generator ──
            # Rayon thread pool is created ONCE at first use and reused for
            # every subsequent call.  reset() repositions to byte 0 — O(µs),
            # no allocation, no thread pool teardown.
            gen = _get_dgen_proc_gen(total_bytes)
            gen.reset()
            bytesview = gen.get_chunk(total_bytes)
        
        # Convert to NumPy array with correct dtype and reshape (ZERO-COPY)
        # np.frombuffer on BytesView is zero-copy because BytesView implements buffer protocol
        arr = np.frombuffer(bytesview, dtype=dtype).reshape(shape)
        
        # Make writable copy only if needed. The read-only view is valid and safe
        # when the caller only reads the array (e.g. np.savez). Pass writeable=False
        # to skip the copy and save one full array allocation.
        if writeable:
            return arr.copy()
        return arr
    
    # Slow path: NumPy random generation (legacy method)
    if rng is None or seed is not None:
        # When a seed is explicitly provided, always create a fresh seeded Generator
        # so that the seed takes effect regardless of what rng was passed by the caller.
        rng = np.random.default_rng(seed=seed)  # seed=None = entropy
    if not np.issubdtype(dtype, np.integer):
        # Only float32 and float64 are supported by rng.random
        if dtype not in (np.float32, np.float64):
            arr = rng.random(size=shape, dtype=np.float32)
            return arr.astype(dtype)
        else:
            return rng.random(size=shape, dtype=dtype)
    
    # For integer dtypes, generate float32 first then scale and cast
    dtype_info = np.iinfo(dtype)
    records = rng.random(size=shape, dtype=np.float32)
    records = records * (dtype_info.max - dtype_info.min) + dtype_info.min
    records = records.astype(dtype)
    return records
