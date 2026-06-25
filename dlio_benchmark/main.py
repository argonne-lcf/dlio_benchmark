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
import math
import subprocess
import time

import numpy as np

# Reduce TF and CUDA logging

import hydra
from omegaconf import DictConfig
from dlio_benchmark.common.enumerations import StorageType as _StorageType


# ---------------------------------------------------------------------------
# Page-cache flush configuration helpers (mlcommons/storage #487).
# ---------------------------------------------------------------------------

#: Default per-call timeout for `sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
#: when DLIO_DROP_CACHES_TIMEOUT is unset, empty, or unparseable. Picked to
#: bound the original mlcommons/storage #391 hang case at a few seconds of
#: cumulative wait across epochs while still completing on most hardware.
_DROP_CACHES_TIMEOUT_DEFAULT_SECONDS = 30


def _resolve_drop_caches_timeout(env=None) -> int:
    """Read the page-cache flush timeout, in seconds, from the environment.

    Behavior is deliberately forgiving: any unset / empty / unparseable /
    sub-1 value collapses to the default. The lower bound matters because
    `subprocess.run(timeout=...)` rejects 0 and negative values at call
    time, and we don't want a typo in an operator's env to crash DLIO.

    Args:
        env: Mapping to read from. Defaults to ``os.environ``. Exposed for
             tests so they don't have to monkey-patch the global env.

    Returns:
        Integer >= 1 representing seconds.
    """
    if env is None:
        env = os.environ
    raw = (env.get("DLIO_DROP_CACHES_TIMEOUT") or "").strip()
    if not raw:
        return _DROP_CACHES_TIMEOUT_DEFAULT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return _DROP_CACHES_TIMEOUT_DEFAULT_SECONDS
    return max(value, 1)


def _apply_settle_guard(args, comm) -> None:
    """Sleep after data generation for eventual-consistency object stores.

    Only activates when *both* conditions are true:
      - ``args.storage_type`` is not ``LOCAL_FS`` (i.e. an object store)
      - ``args.post_generation_settle_seconds > 0``

    Rank-0 sleeps for the configured duration; then all ranks barrier so
    they proceed together.  Default is 0.0 — zero behaviour change for
    existing configs.
    """
    if (args.post_generation_settle_seconds > 0
            and args.storage_type != _StorageType.LOCAL_FS):
        if args.my_rank == 0:
            time.sleep(args.post_generation_settle_seconds)
        comm.barrier()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# Remove PyTorch warning when libtorch_cuda_cu.so isn't found
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dlio_benchmark.checkpointing.checkpointing_factory import CheckpointingFactory
from dlio_benchmark.common.constants import MODULE_DLIO_BENCHMARK
from dlio_benchmark.common.enumerations import DatasetType, MetadataType
from dlio_benchmark.utils.utility import utcnow, DLIOMPI, Profile, dft_ai, DLIOLogger
from dlio_benchmark.utils.statscounter import StatsCounter
from dlio_benchmark.utils.config import LoadConfig, ConfigArguments, GetConfig
from dlio_benchmark.profiler.profiler_factory import ProfilerFactory
from dlio_benchmark.framework.framework_factory import FrameworkFactory
from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
from dlio_benchmark.storage.storage_factory import StorageFactory

dlp = Profile(MODULE_DLIO_BENCHMARK)

dftracer_initialize = True
dftracer_finalize   = True
dftracer            = None

class DLIOBenchmark(object):
    """
    The Benchmark represents the I/O behavior of deep learning applications.
    """

    def __init__(self, cfg):
        """
        This initializes the DLIO benchmark. Intialization includes:
        <ul>
            <li> argument parser </li>
            <li> profiler instances </li>
            <li> internal components </li>
            <li> local variables </li>
        </ul>
        """
        global dftracer, dftracer_initialize, dftracer_finalize

        t0 = time.time()
        self.args = ConfigArguments.get_instance()
        LoadConfig(self.args, cfg)


        self.storage = StorageFactory().get_storage(self.args.storage_type, self.args.storage_root,
                                                    self.args.framework)

        self.output_folder = self.args.output_folder
        os.makedirs(self.args.output_folder, mode=0o755, exist_ok=True)
        self.comm = DLIOMPI.get_instance().comm()
        self.my_rank = self.args.my_rank = DLIOMPI.get_instance().rank()
        self.comm_size = self.args.comm_size = DLIOMPI.get_instance().size()
        self.data_folder = self.args.data_folder
        self.storage_root = self.args.storage_root
        if self.args.storage_root:
            self.storage.create_namespace(exist_ok=True)
        self.framework = FrameworkFactory().get_framework(self.args.framework,
                                                          self.args.do_profiling)

        # Delete previous logfile
        if self.my_rank == 0:
            if os.path.isfile(self.args.logfile_path):
                os.remove(self.args.logfile_path)
        self.comm.barrier()
        # Configure the logging library
        self.args.configure_dlio_logging(is_child=False)
        self.logger = DLIOLogger.get_instance()

        if dftracer_initialize:
            dftracer = self.args.configure_dftracer(is_child=False, use_pid=False)

        if self.my_rank == 0:
            self.logger.output(f"[DEBUG DLIOBenchmark.__init__] After LoadConfig:")
            self.logger.output(f"  storage_type   = {self.args.storage_type!r}")
            self.logger.output(f"  storage_root   = {self.args.storage_root!r}")
            self.logger.output(f"  storage_options= {self.args.storage_options!r}")
            self.logger.output(f"  data_folder    = {self.args.data_folder!r}")
            self.logger.output(f"  framework      = {self.args.framework!r}")
            self.logger.output(f"  num_files_train= {self.args.num_files_train!r}")
            self.logger.output(f"  record_length  = {self.args.record_length!r}")
            self.logger.output(f"  generate_data  = {self.args.generate_data!r}")
            self.logger.output(f"  do_train       = {self.args.do_train!r}")
            self.logger.output(f"  do_checkpoint  = {self.args.do_checkpoint!r}")
            self.logger.output(f"  epochs         = {self.args.epochs!r}")
            self.logger.output(f"  batch_size     = {self.args.batch_size!r}")
        
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_DLIO_BENCHMARK):
            mode = []
            if self.args.generate_data:
                mode += ["Generating data"]
            if self.args.do_train:
                mode += ["Training"]
            if self.args.do_eval:
                mode += ["Evaluation"]
            if self.args.do_checkpoint:
                mode += ["Checkpointing"]
            if self.args.my_rank == 0:
                self.logger.output(f"{utcnow()} Running DLIO [{' & '.join(mode)}] with {self.args.comm_size} process(es)")
                try:
                    self.logger.output(
                        f"{utcnow()} Reading workload YAML config file '{hydra_cfg.runtime.config_sources[1]['path']}/workload/{hydra_cfg.runtime.choices.workload}.yaml'")
                except:
                    pass
            self.generate_only = self.args.generate_only
            self.do_profiling = self.args.do_profiling

            self.data_generator = None
            self.num_files_train = self.args.num_files_train
            self.num_subfolders_train = self.args.num_subfolders_train
            self.num_subfolders_eval = self.args.num_subfolders_eval
            self.num_samples = self.args.num_samples_per_file
            self.total_training_steps = self.args.total_training_steps
            
            self.epochs = self.args.epochs
            self.batch_size = self.args.batch_size
            self.computation_time = self.args.computation_time

            if self.do_profiling:
                self.profiler = ProfilerFactory().get_profiler(self.args.profiler)

            if self.args.generate_data:
                self.data_generator = GeneratorFactory.get_generator(self.args.format)
            # Checkpointing support
            self.do_checkpoint = self.args.do_checkpoint
            self.steps_between_checkpoints = self.args.steps_between_checkpoints
            self.epochs_between_checkpoints = self.args.epochs_between_checkpoints
            self.checkpoint_after_epoch = self.args.checkpoint_after_epoch

            # Evaluation support
            self.do_eval = self.args.do_eval
            self.num_files_eval = self.args.num_files_eval

            self.batch_size_eval = self.args.batch_size_eval
            self.eval_time = self.args.eval_time
            self.eval_after_epoch = self.args.eval_after_epoch
            self.epochs_between_evals = self.args.epochs_between_evals
        self.stats = StatsCounter()

    @dlp.log
    def initialize(self):
        """
        Initializes the benchmark runtime.
        - It generates the required data
        - Start profiling session for Darshan and Tensorboard.
        """
        self.comm.barrier()

        if self.args.generate_data:
            if self.args.my_rank == 0:
                self.logger.output(f"{utcnow()} Starting data generation")
            self.data_generator.generate()
            # important to have this barrier to ensure that the data generation is done for all the ranks
            self.comm.barrier()
            if self.args.my_rank == 0:
                self.logger.output(f"{utcnow()} Generation done")
            _apply_settle_guard(self.args, self.comm)

        if not self.generate_only and self.do_profiling:
            self.profiler.start()
            self.framework.start_framework_profiler()
            self.comm.barrier()
            if self.args.my_rank == 0:
                self.logger.info(f"{utcnow()} Profiling Started with {self.args.profiler}")
        self.comm.barrier()
        file_list_train = []
        file_list_eval = []
        num_subfolders = 0
        if self.args.do_train:
            # ── Streaming rank-0 file listing with round-robin sharding ──
            # Only rank 0 performs directory walks.  Files are streamed in
            # chunks and each rank keeps every comm_size-th file (round-robin).
            # This gives perfectly balanced shards when total % comm_size == 0.
            self.args.files_pre_sharded = True

            for dataset_type in [DatasetType.TRAIN, DatasetType.VALID]:
                t_listing_start = time.time()
                if dataset_type == DatasetType.TRAIN:
                    num_subfolders = self.num_subfolders_train
                else:
                    num_subfolders = self.num_subfolders_eval

                my_files = []
                global_count = 0

                _CHUNK_SIZE = 1_000_000  # max files per bcast to bound rank-0 memory

                def _filter_round_robin(chunk, start_idx):
                    """Keep files where (start_idx + position) % comm_size == my_rank."""
                    for i, fpath in enumerate(chunk):
                        if (start_idx + i) % self.comm_size == self.my_rank:
                            my_files.append(fpath)

                if self.args.skip_listing:
                    # ── Deterministic file list (skip S3 listing entirely) ─
                    # Generate file URIs from DLIO's naming convention without
                    # any storage API calls or MPI communication.  Each rank
                    # independently computes its own round-robin shard.
                    # Convention: {file_prefix}_{index:0N}_of_{total}.{format}
                    # For subfoldered layouts: {subfolder}/{file_prefix}_{index:0N}_of_{total}.{format}
                    # where subfolder = str(index % num_subfolders).zfill(nd_sf)
                    num_files_expected = (
                        self.num_files_train if dataset_type is DatasetType.TRAIN
                        else (self.num_files_eval if self.do_eval else 0)
                    )
                    if num_files_expected > 0:
                        nd_f = len(str(num_files_expected))
                        nd_sf = len(str(max(num_subfolders - 1, 0))) if num_subfolders > 0 else 0
                        for idx in range(self.my_rank, num_files_expected, self.comm_size):
                            fname = f"{self.args.file_prefix}_{str(idx).zfill(nd_f)}_of_{num_files_expected}.{self.args.format}"
                            if num_subfolders > 0:
                                sf = str(idx % num_subfolders).zfill(nd_sf)
                                rel = os.path.join(sf, fname)
                            else:
                                rel = fname
                            uri = self.storage.get_uri(
                                os.path.join(self.args.data_folder, f"{dataset_type}", rel))
                            my_files.append(uri)
                        global_count = num_files_expected
                    # ── Sampling validation (rank 0 only) ─────────────
                    # Confirm the naming convention is correct by checking
                    # that a sample of files actually exists in storage.
                    # Always checks the first and last file, plus every
                    # listing_validation_interval-th file in between.
                    # If any check fails, raises an informative error.
                    if self.my_rank == 0 and num_files_expected > 0 and \
                            self.args.listing_validation_interval > 0:
                        interval = self.args.listing_validation_interval
                        val_indices = sorted(
                            {0, num_files_expected - 1} |
                            set(range(0, num_files_expected, interval))
                        )
                        n_checks = len(val_indices)
                        # ── Header: tell the user what is about to happen ──
                        self.logger.output(
                            f"{utcnow()} skip_listing [{dataset_type}]: validating "
                            f"{n_checks:,} of {num_files_expected:,} files "
                            f"(first, last, every {interval:,}) via HEAD requests ...")
                        failed_uris = []
                        t_val_start = time.time()
                        # Report progress every ~10 % of checks, but at least
                        # every 500 checks and no more often than every 100.
                        progress_stride = max(100, min(500, n_checks // 10))
                        for check_num, vidx in enumerate(val_indices):
                            vfname = f"{self.args.file_prefix}_{str(vidx).zfill(nd_f)}_of_{num_files_expected}.{self.args.format}"
                            if num_subfolders > 0:
                                vsf = str(vidx % num_subfolders).zfill(nd_sf)
                                vrel = os.path.join(vsf, vfname)
                            else:
                                vrel = vfname
                            vuri = self.storage.get_uri(
                                os.path.join(self.args.data_folder, f"{dataset_type}", vrel))
                            if not self.storage.file_exists(vuri):
                                failed_uris.append(vuri)
                            # Periodic progress line (but not on the very first check)
                            if check_num > 0 and check_num % progress_stride == 0:
                                elapsed = time.time() - t_val_start
                                rate = check_num / elapsed if elapsed > 0 else 0
                                pct = 100.0 * check_num / n_checks
                                eta = (n_checks - check_num) / rate if rate > 0 else 0
                                self.logger.output(
                                    f"{utcnow()} skip_listing [{dataset_type}]:   "
                                    f"{check_num:,}/{n_checks:,} checked "
                                    f"({pct:.0f}%)  —  "
                                    f"{rate:.0f} checks/s  —  "
                                    f"ETA {eta:.0f}s  —  "
                                    f"{len(failed_uris)} failed so far")
                        t_val_end = time.time()
                        elapsed_total = t_val_end - t_val_start
                        rate_total = n_checks / elapsed_total if elapsed_total > 0 else 0
                        if failed_uris:
                            sample_shown = failed_uris[:3]
                            raise Exception(
                                f"skip_listing validation failed: {len(failed_uris)} of "
                                f"{n_checks:,} sampled files missing in [{dataset_type}] "
                                f"after {elapsed_total:.1f}s. "
                                f"First failures: {sample_shown}. "
                                f"Ensure data was generated with DLIO's standard naming "
                                f"convention or set skip_listing=False to use directory "
                                f"listing instead.")
                        self.logger.output(
                            f"{utcnow()} skip_listing [{dataset_type}]: validation complete — "
                            f"all {n_checks:,} samples exist "
                            f"({elapsed_total:.1f}s, {rate_total:.0f} checks/s); "
                            f"{len(my_files):,} URIs ready for rank 0 "
                            f"({global_count:,} total across all ranks)")
                    elif self.my_rank == 0:
                        self.logger.output(
                            f"{utcnow()} skip_listing [{dataset_type}]: generated "
                            f"{len(my_files):,} file URIs deterministically "
                            f"({global_count:,} total — validation disabled)")

                elif num_subfolders > 0:
                    # ── Subfoldered layout: stream with chunked bcast ─────
                    subfolder_names = None
                    if self.my_rank == 0:
                        walk_path = os.path.join(self.args.data_folder, f"{dataset_type}")
                        subfolder_names = sorted(self.storage.walk_node(walk_path))
                    subfolder_names = self.comm.bcast(subfolder_names, root=0)

                    if self.my_rank == 0:
                        # Multi-threaded listing of subfolders on rank 0
                        from concurrent.futures import ThreadPoolExecutor

                        def _list_subfolder(sf_name):
                            sf_path = os.path.join(
                                self.args.data_folder, f"{dataset_type}",
                                sf_name, f"*.{self.args.format}")
                            return self.storage.walk_node(sf_path, use_pattern=True)

                        pending = []
                        listing_threads = self.args.listing_threads
                        with ThreadPoolExecutor(max_workers=listing_threads) as pool:
                            for sf_files in pool.map(_list_subfolder, subfolder_names):
                                pending.extend(sf_files)
                                # Flush in chunks of _CHUNK_SIZE
                                while len(pending) >= _CHUNK_SIZE:
                                    chunk = sorted(pending[:_CHUNK_SIZE])
                                    pending = pending[_CHUNK_SIZE:]
                                    chunk = self.comm.bcast(chunk, root=0)
                                    _filter_round_robin(chunk, global_count)
                                    global_count += len(chunk)
                                    del chunk

                        # Flush remaining
                        if pending:
                            chunk = sorted(pending)
                            pending = []
                            chunk = self.comm.bcast(chunk, root=0)
                            _filter_round_robin(chunk, global_count)
                            global_count += len(chunk)
                            del chunk
                        # Signal end: broadcast empty list
                        self.comm.bcast([], root=0)
                    else:
                        # Non-root ranks: receive chunks until empty sentinel
                        while True:
                            chunk = self.comm.bcast(None, root=0)
                            if not chunk:
                                break
                            _filter_round_robin(chunk, global_count)
                            global_count += len(chunk)
                            del chunk

                else:
                    # ── Flat layout: stream in chunks of _CHUNK_SIZE ──────
                    if self.my_rank == 0:
                        walk_path = os.path.join(self.args.data_folder, f"{dataset_type}")
                        filenames = self.storage.walk_node(walk_path)
                        pending = sorted([
                            self.storage.get_uri(
                                os.path.join(self.args.data_folder, f"{dataset_type}", entry))
                            for entry in filenames
                            if entry.endswith(f'{self.args.format}')
                        ])
                        # Send in chunks
                        for i in range(0, len(pending), _CHUNK_SIZE):
                            chunk = pending[i:i + _CHUNK_SIZE]
                            chunk = self.comm.bcast(chunk, root=0)
                            _filter_round_robin(chunk, global_count)
                            global_count += len(chunk)
                            del chunk
                        del pending
                        # Signal end
                        self.comm.bcast([], root=0)
                    else:
                        while True:
                            chunk = self.comm.bcast(None, root=0)
                            if not chunk:
                                break
                            _filter_round_robin(chunk, global_count)
                            global_count += len(chunk)
                            del chunk

                # ── Validation ───────────────────────────────────────────
                if dataset_type is DatasetType.TRAIN:
                    expected = self.num_files_train
                else:
                    expected = self.num_files_eval if self.do_eval else 0

                if not self.generate_only and expected > global_count:
                    raise Exception(
                        "Not enough dataset is found; Please run the code with "
                        "++workload.workflow.generate_data=True")

                # Floor-division: ensure every rank has the same file count.
                # Round-robin gives ranks 0..r-1 one extra file; trim to floor.
                effective = min(expected, global_count) if expected > 0 else global_count
                files_per_rank = effective // self.comm_size
                my_files = my_files[:files_per_rank]

                if dataset_type is DatasetType.TRAIN:
                    file_list_train = my_files
                    global_train_count = global_count
                elif dataset_type is DatasetType.VALID:
                    file_list_eval = my_files

                t_listing_end = time.time()
                if self.my_rank == 0:
                    self.logger.output(
                        f"{utcnow()} File listing [{dataset_type}]: "
                        f"{global_count} files discovered, {len(my_files)} assigned to rank 0, "
                        f"completed in {t_listing_end - t_listing_start:.2f}s")

            if self.my_rank == 0:
                self.logger.output(
                    f"{utcnow()} Streamed file sharding: {global_train_count} train files "
                    f"across {self.comm_size} ranks via round-robin "
                    f"(rank 0 shard: {len(file_list_train)} files)")

        self.args.derive_configurations(file_list_train, file_list_eval)
        self.args.validate()
        self.checkpointing_mechanism = None
        self.stats.checkpoint_size = 0
        if (not self.generate_only) and (self.do_checkpoint):
            self.checkpointing_mechanism = CheckpointingFactory().get_mechanism(self.args.checkpoint_mechanism)
            self.stats.checkpoint_size = self.checkpointing_mechanism.checkpoint_size    
        self.comm.barrier()

    @dft_ai.pipeline.evaluate
    def _eval(self, epoch):
        """
        Evaluation loop will read a separate dataset and has its own own computation time.
        """
        step = 1
        if self.args.files_pre_sharded:
            total = self.args.eval_steps  # agreed via allreduce(MIN)
        else:
            total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
        loader = self.framework.get_loader(DatasetType.VALID)
        self.stats.start_loading()
        for batch in loader.next():
            # @ray: fixing uneven data fetch and computation count (same issue with `_train` below)
            # Check if max steps reached to prevent incomplete fetch/compute pairs
            # This ensures accurate event counting by stopping compute when step limit is hit
            if step > total:
                break
            self.stats.eval_batch_loaded(epoch, step)
            eval_time = self.eval_time
            self.stats.start_compute()
            self.framework.compute(batch, epoch, step, eval_time)
            self.stats.eval_batch_processed(epoch, step)
            step += 1
            self.stats.start_loading()
        return step - 1

    @dlp.log
    def _checkpoint(self):
        """
        Checkpointing loop will save the checkpoint after a certain number of steps.
        """
        self.stats.start_epoch()
        if self.args.num_checkpoints_write > 0:
            self._checkpoint_write()
        num_checkpoints_exists = len(self.storage.walk_node(self.args.checkpoint_folder))
        if num_checkpoints_exists < self.args.num_checkpoints_read:
            raise Exception("Number of checkpoints to be read: {self.args.num_checkpoints_read} is more than the number of checkpoints available: {num_checkpoints_exists}")
        if self.args.num_checkpoints_read > 0:
            self._checkpoint_read()
        self.stats.end_epoch()

    @dlp.log
    def _checkpoint_write(self):
        if self.comm.rank == 0:
            self.logger.output(f"{utcnow()} Checkpointing write started")
        block = 1  # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1  # Steps are taken within blocks
        epoch = 1
        for i in range(self.args.num_checkpoints_write):
            #self.stats.start_block(epoch, block)
            # We still make sure that the checkpoint is done after allreduce; therefore, allreduce here is required. 
            self.framework.compute(None, epoch, block_step, self.args.time_between_checkpoints)
            self.comm.barrier()
            self.stats.start_save_ckpt(epoch, block, overall_step)
            self.checkpointing_mechanism.save_checkpoint(epoch, overall_step)
            if self.args.checkpoint_rank_sync: 
                self.comm.barrier()
            self.stats.end_save_ckpt(epoch, block)
            block = block+1
            overall_step = overall_step + 1
        if self.comm.rank == 0:
            self.logger.output(f"{utcnow()} Checkpointing write finished")

    @dlp.log
    def _checkpoint_read(self):
        if self.comm.rank == 0:
            self.logger.output(f"{utcnow()} Checkpointing read started")
        block = 1  # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1  # Steps are taken within blocks
        epoch = 1
        for i in range(self.args.num_checkpoints_read):
            self.framework.compute(None, epoch, block_step, self.args.time_between_checkpoints)
            self.comm.barrier()
            self.stats.start_load_ckpt(epoch, block, overall_step)
            self.checkpointing_mechanism.load_checkpoint(epoch, overall_step)
            if self.args.checkpoint_rank_sync: 
                self.comm.barrier()
            self.stats.end_load_ckpt(epoch, block)
            block = block+1
            overall_step = overall_step + 1
        if self.comm.rank == 0:
            self.logger.output(f"{utcnow()} Checkpointing write started")

    @dft_ai.pipeline.train
    def _train(self, epoch):
        """
        Training loop for reading the dataset and performing training computations.
        :return: returns total steps.
        """
        block = 1  # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1  # Steps are taken within blocks
        if self.args.files_pre_sharded:
            max_steps = self.args.training_steps  # agreed via allreduce(MIN)
        else:
            max_steps = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
        self.steps_per_epoch = max_steps
        # Start the very first block
        self.stats.start_block(epoch, block)
        loader = self.framework.get_loader(dataset_type=DatasetType.TRAIN)
        self.stats.start_loading()
        for batch in loader.next():
            # @ray: fixing uneven data fetch and computation count
            # Check if max steps reached to prevent incomplete fetch/compute pairs
            # This ensures accurate event counting by stopping compute when step limit is hit
            if overall_step > max_steps or ((self.total_training_steps > 0) and (overall_step > self.total_training_steps)):
                if self.args.my_rank == 0:
                    self.logger.info(f"{utcnow()} Maximum number of steps reached")
                break
            self.stats.batch_loaded(epoch, overall_step, block)
            computation_time = self.args.computation_time
            if (isinstance(computation_time, dict) and len(computation_time) > 0) or (isinstance(computation_time, float) and  computation_time > 0):
                self.framework.trace_object("Train", overall_step, 1)
            self.stats.start_compute()
            self.framework.compute(batch, epoch, block_step, self.computation_time)
            self.stats.batch_processed(epoch, overall_step, block)
            # This is the barrier to simulate allreduce. It is required to simulate the actual workloads.
            self.comm.barrier()
            if self.do_checkpoint and (
                    self.steps_between_checkpoints >= 0) and overall_step == self.next_checkpoint_step:
                self.stats.end_block(epoch, block, block_step)
                self.stats.start_save_ckpt(epoch, block, overall_step)
                self.checkpointing_mechanism.save_checkpoint(epoch, overall_step)
                self.stats.end_save_ckpt(epoch, block)
                block += 1
                # Reset the number of steps after every checkpoint to mark the start of a new block
                block_step = 1
                self.next_checkpoint_step += self.steps_between_checkpoints
            else:
                block_step += 1
            overall_step += 1
            # start a new block here
            if block_step == 1 and block != 1:
                self.stats.start_block(epoch, block)
            self.stats.start_loading()

        # Always closes the current block. It is safe to call end_block for already ended block, as there's a guard inside.
        self.stats.end_block(epoch, block, block_step - 1)

        self.comm.barrier()
        if self.do_checkpoint and (self.steps_between_checkpoints < 0) and (epoch == self.next_checkpoint_epoch):
            self.stats.start_save_ckpt(epoch, block, overall_step-1)
            self.checkpointing_mechanism.save_checkpoint(epoch, overall_step)
            self.stats.end_save_ckpt(epoch, block)
            self.next_checkpoint_epoch += self.epochs_between_checkpoints
        return overall_step

    @dft_ai
    def run(self):
        """
        Run the total epochs for training. 
        On each epoch, it prepares dataset for reading, it trains, and finalizes the dataset.
        If evaluation is enabled, it reads the eval dataset, performs evaluation and finalizes.
        """
        self.stats.start_run()
        if (not self.generate_only) and (not self.args.checkpoint_only):
            # Print out the expected number of steps for each epoch and evaluation
            if self.my_rank == 0:
                if self.args.files_pre_sharded:
                    total = math.floor(self.num_samples * self.num_files_train / self.batch_size)
                    self.logger.output(
                        f"{utcnow()} Max steps per epoch per rank: {total} = {self.num_samples} * {self.num_files_train} / {self.batch_size} (samples per file * local files / batch size)")
                else:
                    total = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
                    self.logger.output(
                        f"{utcnow()} Max steps per epoch: {total} = {self.num_samples} * {self.num_files_train} / {self.batch_size} / {self.comm_size} (samples per file * num files / batch size / comm size)")
                if self.total_training_steps > 0:
                    self.logger.output(
                        f"{utcnow()} Total training steps is set to be {self.total_training_steps}. Will only run up to {min(total*self.args.epochs, self.total_training_steps)}"
                    )
                if self.do_eval:
                    if self.args.files_pre_sharded:
                        total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval)
                        self.logger.output(
                            f"{utcnow()} Steps per eval per rank: {total} = {self.num_samples} * {self.num_files_eval} / {self.batch_size_eval} (samples per file * local files / batch size eval)")
                    else:
                        total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
                        self.logger.output(
                            f"{utcnow()} Steps per eval: {total} = {self.num_samples} * {self.num_files_eval} / {self.batch_size_eval} / {self.comm_size} (samples per file * num files / batch size eval / comm size)")

            # Keep track of the next epoch at which we will evaluate
            next_eval_epoch = self.eval_after_epoch
            self.next_checkpoint_epoch = self.checkpoint_after_epoch
            epoch = 1
            # Initialize the dataset
            self.args.reconfigure(epoch)
            self.framework.init_loader(self.args.format, epoch=epoch, data_loader=self.args.data_loader)
            self.framework.get_loader(dataset_type=DatasetType.TRAIN).read()
            if self.do_eval:
                self.framework.get_loader(dataset_type=DatasetType.VALID).read()

            # Pre-warm workers: trigger DataLoader worker spawn before epoch 1.
            # Without persistent_workers, workers re-spawn on each iter() call.
            # This pre-warm ensures the first epoch doesn't include spawn latency.
            train_loader = self.framework.get_loader(dataset_type=DatasetType.TRAIN)
            if hasattr(train_loader, '_dataset') and train_loader._dataset is not None:
                warmup_iter = iter(train_loader._dataset)
                try:
                    next(warmup_iter)
                except StopIteration:
                    pass
                del warmup_iter
                if self.my_rank == 0:
                    self.logger.output(f"{utcnow()} Worker pre-warm complete ({self.args.read_threads} workers spawned)")

            self.comm.barrier()
            # The flush has two distinct failure modes (mlcommons/storage #391, #487):
            #
            #   * sudo -n refuses (no NOPASSWD configured, or sudo missing)
            #       -> non-zero exit code, fast.  Warn once and disable for the
            #          run so we don't pay the failure cost every epoch.  This
            #          is what #391 originally fixed (the interactive sudo
            #          prompt that hung for ~16 hours).
            #
            #   * sudo -n authenticated, but the kernel itself is slow
            #       -> subprocess.TimeoutExpired.  Don't disable; the kernel is
            #          working, just slowly.  The next epoch retries.
            #
            # The per-call timeout is overridable via DLIO_DROP_CACHES_TIMEOUT
            # so large-RAM hosts can raise the ceiling without an upstream change.
            drop_caches_timeout = _resolve_drop_caches_timeout()
            drop_caches_disabled = False
            drop_caches_timeout_count = 0
            drop_caches_attempt_count = 0
            for epoch in dft_ai.pipeline.epoch.iter(range(1, self.epochs + 1), include_iter=False):
                # Flush page cache before each epoch so reads bypass the OS buffer cache.
                # Rank 0 does the flush via sudo -n (non-interactive); all ranks barrier-
                # wait so no rank starts reading stale cached data.
                if self.my_rank == 0 and not drop_caches_disabled:
                    drop_caches_attempt_count += 1
                    try:
                        subprocess.run(
                            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                            check=True, timeout=drop_caches_timeout,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                    except subprocess.TimeoutExpired:
                        # sudo -n already authenticated (otherwise we'd see a
                        # quick non-zero exit, not a timeout). The kernel is
                        # the slow path. Don't disable — the next epoch retries.
                        #
                        # Warn on every timeout (mlcommons/storage #487 reopen):
                        # the original warn-once-then-silent behavior left
                        # operators unable to tell whether subsequent retries
                        # succeeded or kept timing out at exactly the ceiling.
                        # First occurrence is verbose with the remediation hint;
                        # later occurrences are a one-liner so the log stays
                        # scannable while every retry is still surfaced.
                        drop_caches_timeout_count += 1
                        if drop_caches_timeout_count == 1:
                            self.logger.warning(
                                f"Page cache flush did not finish within "
                                f"{drop_caches_timeout}s (epoch {epoch}). The next "
                                f"epoch will retry. If this recurs, raise the ceiling "
                                f"with DLIO_DROP_CACHES_TIMEOUT=<seconds> (e.g. 300)."
                            )
                        else:
                            self.logger.warning(
                                f"Page cache flush timed out again at "
                                f"{drop_caches_timeout}s (epoch {epoch}); "
                                f"see earlier warning for remediation."
                            )
                    except Exception as exc:
                        drop_caches_disabled = True
                        self.logger.warning(
                            f"Could not flush page cache between epochs: {exc}. "
                            "Per-epoch reads may be served from the OS buffer cache, "
                            "inflating throughput numbers. To enable, configure "
                            "passwordless sudo for `sh -c 'echo 3 > /proc/sys/vm/"
                            "drop_caches'`."
                        )
                self.comm.barrier()
                self.stats.start_epoch(epoch)
                self.next_checkpoint_step = self.steps_between_checkpoints
                self.stats.start_train(epoch)
                steps = self._train(epoch)
                self.stats.end_train(epoch, steps)
                self.logger.debug(f"{utcnow()} Rank {self.my_rank} returned after {steps} steps.")
                self.framework.get_loader(DatasetType.TRAIN).finalize()
                # Perform evaluation if enabled
                if self.do_eval and epoch >= next_eval_epoch:
                    next_eval_epoch += self.epochs_between_evals
                    self.stats.start_eval(epoch)
                    self._eval(epoch)
                    self.stats.end_eval(epoch)
                    self.framework.get_loader(DatasetType.VALID).finalize()
                self.args.reconfigure(epoch + 1) # reconfigure once per epoch
                # Refresh serialized args so next epoch's workers see resharded file list
                train_loader = self.framework.get_loader(dataset_type=DatasetType.TRAIN)
                if hasattr(train_loader, 'refresh_args'):
                    train_loader.refresh_args()
                if self.do_eval:
                    eval_loader = self.framework.get_loader(dataset_type=DatasetType.VALID)
                    if hasattr(eval_loader, 'refresh_args'):
                        eval_loader.refresh_args()
                # Pre-warm workers for next epoch (spawn + init outside timed window)
                if hasattr(train_loader, '_dataset') and train_loader._dataset is not None:
                    warmup_iter = iter(train_loader._dataset)
                    try:
                        next(warmup_iter)
                    except StopIteration:
                        pass
                    del warmup_iter
                    if self.my_rank == 0:
                        self.logger.output(f"{utcnow()} Worker pre-warm complete for epoch {epoch + 1} ({self.args.read_threads} workers spawned)")
                self.stats.end_epoch(epoch)

            # End-of-run page-cache-flush summary (mlcommons/storage #487 reopen).
            # Emit only when at least one timeout occurred so quiet runs stay
            # quiet. Lets the operator confirm at a glance how many epochs were
            # affected without grepping for per-epoch warnings.
            if self.my_rank == 0 and drop_caches_timeout_count > 0:
                self.logger.warning(
                    f"Page cache flush timed out in {drop_caches_timeout_count} of "
                    f"{drop_caches_attempt_count} epochs. Reads in those epochs may "
                    f"have been served from the OS buffer cache; consider raising "
                    f"DLIO_DROP_CACHES_TIMEOUT for future runs."
                )

        if (self.args.checkpoint_only):
            self._checkpoint()            
        self.stats.end_run()

    @dlp.log
    def finalize(self):
        """
        It finalizes the dataset once training is completed.
        """
        global dftracer, dftracer_initialize, dftracer_finalize

        self.comm.barrier()

        if self.checkpointing_mechanism:
            self.checkpointing_mechanism.finalize()
        if not self.generate_only:
            if self.do_profiling:
                self.profiler.stop()
                self.framework.stop_framework_profiler()
                self.comm.barrier()
                if self.my_rank == 0:
                    self.logger.info(f"{utcnow()} Profiling stopped")
            if not self.args.keep_files:
                self.logger.info(f"{utcnow()} Keep files set to False. Deleting dataset")
                self.comm.barrier()
                if self.my_rank == 0:
                    if self.storage.get_node(self.args.data_folder):
                        self.storage.delete_node(self.args.data_folder)
                        self.logger.info(f"{utcnow()} Deleted data files")

            # Save collected stats to disk
            self.stats.finalize()
            self.stats.save_data()
        self.comm.barrier()
        if dftracer_finalize and dftracer:
            self.args.finalize_dftracer(dftracer)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_benchmark(cfg: DictConfig):    
    benchmark = DLIOBenchmark(cfg['workload'])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()

def set_dftracer_initialize(status):
    global dftracer, dftracer_initialize, dftracer_finalize
    dftracer_initialize = status

def set_dftracer_finalize(status):
    global dftracer, dftracer_initialize, dftracer_finalize
    dftracer_finalize = status

def main() -> None:
    """
    The main method to start the benchmark runtime.
    """
    DLIOMPI.get_instance().initialize()
    run_benchmark()
    DLIOMPI.get_instance().finalize()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def query_config(cfg: DictConfig):
    DLIOMPI.get_instance().initialize()
    config = cfg['workload']
    
    value = None
    if "query" in config["workflow"]:
        key = config["workflow"]["query"]
        args = ConfigArguments.get_instance()
        LoadConfig(args, config)
        value = GetConfig(args, key)
    print(value) if value else print("None")
    DLIOMPI.get_instance().finalize()
    
if __name__ == '__main__':
    main()
    exit(0)
