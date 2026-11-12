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

Regression tests for reader.file_shuffle with the index sampler (PyTorch loader).

The two shuffle options do different things:

* file_shuffle permutes the file list, the same way on every rank, before the
  samples are split between the ranks. It decides which files a rank reads and
  in which order; the samples of a file are still read one after the other.
* sample_shuffle permutes the order in which a rank reads its own samples
  (issue #360). It does not change which samples a rank reads.

The PyTorch loader builds its readers once, before the first epoch, and they keep
the file order of that epoch. The tests therefore check every epoch for a
complete, disjoint split, but do not expect the file order to change between
epochs. Once sample_shuffle works with this loader (issue #360), it is the
sample order that changes every epoch; the test of the two roles is skipped
until then.

The run tests read an indexed_binary dataset with several samples per file
through a reader that logs every read made on the data files
(tests/file_shuffle_reader.py), so they check the order in which files were
really read, per rank and per epoch.

Run with several ranks, for example:
    mpirun -np 2 pytest -v tests/dlio_file_shuffle_test.py
    mpirun -np 3 --oversubscribe pytest -v tests/dlio_file_shuffle_test.py
"""
import collections
import gc
import glob
import os
import shutil

import numpy as np
import pytest
from hydra import initialize_config_dir, compose
from mpi4py import MPI
from omegaconf import OmegaConf

import dlio_benchmark
from dlio_benchmark.common.enumerations import DataLoaderSampler, Shuffle
from dlio_benchmark.utils.config import ConfigArguments, LoadConfig
from dlio_benchmark.utils.utility import DLIOMPI
from tests.file_shuffle_reader import READ_LOG_ENV, READ_LOG_COLUMNS
from tests.utils import TEST_TIMEOUT_SECONDS

comm = MPI.COMM_WORLD
config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

SAMPLES_PER_FILE = 4
BATCH_SIZE = 2
FILES_PER_RANK = 8
# a multiple of samples_per_file * batch_size * ranks, so main.py does not trim the file list
NUM_FILES = FILES_PER_RANK * comm.size
SAMPLES_PER_RANK = FILES_PER_RANK * SAMPLES_PER_FILE
EPOCHS = 2
DATA_FOLDER = "data/file_shuffle"
READ_LOGS = "data/file_shuffle_reads"
READER = "tests.file_shuffle_reader.ReadRecordingIndexedBinaryReader"


def init():
    # A PyTorch run with read_threads=0 unpickles the configuration in this process, which leaves
    # DLIOMPI in its "child" state; start from a fresh, initialized instance (collective call).
    DLIOMPI.reset()
    DLIOMPI.get_instance().initialize()


def clean():
    comm.Barrier()
    if comm.rank == 0:
        shutil.rmtree(DATA_FOLDER, ignore_errors=True)
        shutil.rmtree(READ_LOGS, ignore_errors=True)
        shutil.rmtree("output", ignore_errors=True)
    comm.Barrier()


# ---------------------------------------------------------------------------
# helpers shared by the checks
# ---------------------------------------------------------------------------

def _file_order(reads):
    """Files in the order they were first read."""
    order = []
    for path, _ in reads:
        if not order or order[-1] != path:
            if path in order:
                return None  # a file was left and read again: samples of files are interleaved
            order.append(path)
    return order


def _read_file_by_file(reads):
    """True when each file is read whole, front to back, before the next file starts."""
    order = _file_order(reads)
    return order is not None and reads == [(f, s) for f in order for s in range(SAMPLES_PER_FILE)]


def _assert_partition(per_rank, files, what):
    """Every (file, sample) of the dataset is read by exactly one rank."""
    counts = collections.Counter(pair for reads in per_rank for pair in reads)
    expected = {(f, s): 1 for f in files for s in range(SAMPLES_PER_FILE)}
    duplicated = sorted(p for p, n in counts.items() if n > 1)
    missing = sorted(set(expected) - set(counts))
    assert counts == expected, f"{what}: samples not split between ranks: duplicated {duplicated[:4]}, missing {missing[:4]}"


def _contiguous_block(files, rank):
    return files[rank * FILES_PER_RANK:(rank + 1) * FILES_PER_RANK]


def _short(path):
    return os.path.basename(path).split(".")[0]


# ---------------------------------------------------------------------------
# configuration: file_shuffle has to become a Shuffle value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("yaml_value, expected", [("off", Shuffle.OFF),       # YAML 1.1 reads a bare off as False
                                                  ("'off'", Shuffle.OFF),
                                                  ("seed", Shuffle.SEED),
                                                  ("random", Shuffle.RANDOM)])
def test_file_shuffle_config_value(yaml_value, expected) -> None:
    init()
    ConfigArguments.reset()
    args = ConfigArguments.get_instance()
    LoadConfig(args, OmegaConf.create(f"reader:\n  file_shuffle: {yaml_value}\n"))
    assert args.file_shuffle is expected


# ---------------------------------------------------------------------------
# index map built by ConfigArguments.reconfigure() (no data, no PyTorch)
# ---------------------------------------------------------------------------

FAKE_FILES = [f"/dlio/train/img_{i:02d}_of_{NUM_FILES}.indexed_binary" for i in range(NUM_FILES)]


def _index_maps(file_shuffle, seed=123):
    """Build the index-sampler map with the real reconfigure(1); return the reads it maps for every rank."""
    ConfigArguments.reset()
    args = ConfigArguments.get_instance()
    args.my_rank, args.comm_size = comm.rank, comm.size
    args.data_loader_sampler = DataLoaderSampler.INDEX
    args.file_shuffle, args.sample_shuffle = file_shuffle, Shuffle.OFF
    args.seed, args.seed_change_epoch = seed, True
    args.num_samples_per_file = SAMPLES_PER_FILE
    args.file_list_train, args.file_list_eval = list(FAKE_FILES), []
    total = SAMPLES_PER_FILE * NUM_FILES
    args.total_samples_train, args.total_samples_eval = total, 0
    args.train_sample_index_sum, args.eval_sample_index_sum = total * (total - 1) // 2, 0
    args.reconfigure(1)
    # without sample_shuffle the sampler asks for the global sample indices of the rank in ascending order
    mine = [args.train_global_index_map[g] for g in sorted(args.train_global_index_map)]
    return comm.allgather(mine)


def test_index_map_file_shuffle() -> None:
    init()
    off = _index_maps(Shuffle.OFF)
    for rank, reads in enumerate(off):
        assert reads == [(f, s) for f in _contiguous_block(FAKE_FILES, rank) for s in range(SAMPLES_PER_FILE)]

    for mode in (Shuffle.SEED, Shuffle.RANDOM):
        shuffled = _index_maps(mode)
        _assert_partition(shuffled, FAKE_FILES, f"file_shuffle={mode}")
        assert all(_read_file_by_file(reads) for reads in shuffled), "file_shuffle must not split files"
        moved = [_file_order(r) for r in shuffled] != [_file_order(r) for r in off]
        assert moved, f"file_shuffle={mode} did not change the files read by the ranks"

    seeded = _index_maps(Shuffle.SEED)
    assert _index_maps(Shuffle.SEED) == seeded, "same seed must give the same file order"
    assert _index_maps(Shuffle.SEED, seed=7) != seeded, "another seed must give another file order"


# ---------------------------------------------------------------------------
# PyTorch runs: what the reader really read
# ---------------------------------------------------------------------------

_dftracer_started = False


def _run(name, *, file_shuffle="off", sample_shuffle="off", seed=123, read_threads=1, context="fork",
         generate=False):
    """Train EPOCHS epochs with the PyTorch loader; return {rank: [reads of epoch 1, reads of epoch 2, ...]}."""
    global _dftracer_started
    from dlio_benchmark.main import DLIOBenchmark, set_dftracer_initialize, set_dftracer_finalize
    # several benchmarks run in this process: start dftracer once and never finalize it in between
    set_dftracer_initialize(not _dftracer_started)
    set_dftracer_finalize(False)
    _dftracer_started = True
    log_dir = os.path.abspath(os.path.join(READ_LOGS, name))
    comm.Barrier()
    if comm.rank == 0:
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir)
    comm.Barrier()
    overrides = ['++workload.workflow.generate_data=' + str(generate),
                 '++workload.workflow.train=True',
                 '++workload.workflow.evaluation=False',
                 '++workload.framework=pytorch',
                 '++workload.reader.data_loader=pytorch',
                 '++workload.dataset.format=indexed_binary',
                 f'++workload.dataset.data_folder={DATA_FOLDER}',
                 f'++workload.dataset.num_files_train={NUM_FILES}',
                 f'++workload.dataset.num_samples_per_file={SAMPLES_PER_FILE}',
                 '++workload.dataset.num_subfolders_train=0',
                 '++workload.dataset.num_subfolders_eval=0',
                 '++workload.dataset.record_length_bytes=1024',
                 f'++workload.reader.batch_size={BATCH_SIZE}',
                 f'++workload.reader.read_threads={read_threads}',
                 f'++workload.reader.multiprocessing_context={context}',
                 f'++workload.reader.file_shuffle={file_shuffle}',
                 f'++workload.reader.sample_shuffle={sample_shuffle}',
                 f'++workload.reader.reader_classname={READER}',
                 f'++workload.train.epochs={EPOCHS}',
                 f'++workload.train.seed={seed}',
                 '++workload.train.computation_time=0']
    os.environ[READ_LOG_ENV] = log_dir
    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=overrides)
            ConfigArguments.reset()
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()
            benchmark.run()
            benchmark.finalize()
            del benchmark
            gc.collect()
    finally:
        del os.environ[READ_LOG_ENV]
        init()
    comm.Barrier()
    return _load_reads(log_dir)


def _load_reads(log_dir):
    rows = []
    for log in glob.glob(os.path.join(log_dir, "reads.*.tsv")):
        with open(log) as f:
            rows += [dict(zip(READ_LOG_COLUMNS, line.rstrip("\n").split("\t"))) for line in f]
    by_rank = collections.defaultdict(list)
    for row in rows:
        if row["dataset_type"] == "train":
            by_rank[int(row["rank"])].append(row)
    runs = {}
    for rank in range(comm.size):
        reads = sorted(by_rank[rank], key=lambda r: (int(r["t_ns"]), int(r["pid"]), int(r["seq"])))
        assert len(reads) == EPOCHS * SAMPLES_PER_RANK, \
            f"rank {rank} read {len(reads)} samples, expected {EPOCHS} x {SAMPLES_PER_RANK}"
        _assert_bytes_read(reads)
        pairs = [(r["path"], int(r["sample_index"])) for r in reads]
        runs[rank] = [pairs[e * SAMPLES_PER_RANK:(e + 1) * SAMPLES_PER_RANK] for e in range(EPOCHS)]
    return runs


_index_cache = {}


def _assert_bytes_read(reads):
    """The bytes that went through the file handle are the ones of the logged sample."""
    for r in reads:
        path, sample = r["path"], int(r["sample_index"])
        if path not in _index_cache:
            _index_cache[path] = (np.fromfile(path + ".off.idx", dtype=np.int64, count=SAMPLES_PER_FILE),
                                  np.fromfile(path + ".sz.idx", dtype=np.int64, count=SAMPLES_PER_FILE))
        offsets, sizes = _index_cache[path]
        assert (int(r["offset"]), int(r["nbytes"])) == (offsets[sample], sizes[sample]), \
            f"read of {path} sample {sample} went to offset {r['offset']} ({r['nbytes']} bytes)"


def _dataset_files():
    return sorted(os.path.abspath(f) for f in glob.glob(os.path.join(DATA_FOLDER, "train", "*.indexed_binary")))


def _log_order(title, runs):
    """Log the files each rank read, in read order (sample order when files are interleaved)."""
    if comm.rank != 0:
        return
    for rank, epochs in runs.items():
        for epoch, reads in enumerate(epochs, start=1):
            order = _file_order(reads)
            shown = "files " + " ".join(_short(f)[4:6] for f in order) if order else "samples " + " ".join(
                f"{_short(f)[4:6]}#{s}" for f, s in reads[:10]) + " ..."
            # printed rather than logged: the benchmark reconfigures the logging module
            print(f"[file_shuffle] {title:26s} rank {rank} epoch {epoch}: {shown}", flush=True)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("read_threads, context", [(0, "fork"), (1, "fork"), (1, "spawn")])
def test_torch_file_shuffle_read_order(read_threads, context) -> None:
    """file_shuffle changes which files each rank reads and in which order, every epoch keeps the split."""
    init()
    clean()
    off = _run("off", read_threads=read_threads, context=context, generate=True)
    files = _dataset_files()
    assert len(files) == NUM_FILES
    _log_order("file_shuffle=off", off)
    for rank, epochs in off.items():
        for reads in epochs:
            in_file_order = reads == [(f, s) for f in _contiguous_block(files, rank) for s in range(SAMPLES_PER_FILE)]
            assert in_file_order, f"file_shuffle=off: rank {rank} must read its own block of files in file order"

    for mode in ("seed", "random"):
        shuffled = _run(mode, file_shuffle=mode, read_threads=read_threads, context=context)
        _log_order(f"file_shuffle={mode}", shuffled)
        for epoch in range(EPOCHS):
            _assert_partition([shuffled[r][epoch] for r in range(comm.size)], files,
                              f"file_shuffle={mode} epoch {epoch + 1}")
        for rank, epochs in shuffled.items():
            for reads in epochs:
                assert _read_file_by_file(reads), \
                    f"file_shuffle={mode} rank {rank}: samples of a file are not read one after the other"
        moved = [_file_order(shuffled[r][0]) for r in range(comm.size)] != \
                [_file_order(off[r][0]) for r in range(comm.size)]
        assert moved, f"file_shuffle={mode} did not change the order files are read in"
    clean()


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_torch_file_shuffle_seed_reproducible() -> None:
    """The same seed reads the files in the same order; another seed in another order."""
    init()
    clean()
    first = _run("seed123_a", file_shuffle="seed", seed=123, generate=True)
    again = _run("seed123_b", file_shuffle="seed", seed=123)
    other = _run("seed7", file_shuffle="seed", seed=7)
    _log_order("file_shuffle=seed seed=123", first)
    _log_order("file_shuffle=seed seed=7", other)
    same_seed_same_reads = again == first
    other_seed_same_reads = other == first
    assert same_seed_same_reads, "two runs with seed=123 read the samples in a different order"
    assert not other_seed_same_reads, "seed=7 and seed=123 read the samples in the same order"
    clean()


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_torch_file_and_sample_shuffle_roles() -> None:
    """sample_shuffle reorders a rank's own samples every epoch; file_shuffle picks the rank's files."""
    init()
    clean()
    sample_only = _run("sample", sample_shuffle="seed", generate=True)
    if all(_read_file_by_file(reads) for epochs in sample_only.values() for reads in epochs):
        clean()
        pytest.skip("sample_shuffle does not change the read order of the PyTorch loader (issue #360)")
    files = _dataset_files()
    file_only = _run("file", file_shuffle="seed")
    both = _run("both", file_shuffle="seed", sample_shuffle="seed")
    _log_order("sample_shuffle=seed", sample_only)
    _log_order("file_shuffle=seed", file_only)
    _log_order("file+sample_shuffle=seed", both)

    moved = [_file_order(file_only[r][0]) for r in range(comm.size)] != \
            [_contiguous_block(files, r) for r in range(comm.size)]
    assert moved, "file_shuffle did not change the files read"
    for rank in range(comm.size):
        block = {(f, s) for f in _contiguous_block(files, rank) for s in range(SAMPLES_PER_FILE)}
        assigned = set(file_only[rank][0])
        for epoch in range(EPOCHS):
            # sample_shuffle alone keeps the contiguous split and only changes the order
            assert set(sample_only[rank][epoch]) == block, "sample_shuffle must not change which samples a rank reads"
            assert not _read_file_by_file(sample_only[rank][epoch]), "sample_shuffle did not reorder the samples"
            # with both, the files are the ones file_shuffle gave this rank, read in a shuffled order
            assert set(both[rank][epoch]) == assigned, "sample_shuffle must not change the files file_shuffle chose"
            assert not _read_file_by_file(both[rank][epoch]), "sample_shuffle did not reorder the samples"
        assert sample_only[rank][0] != sample_only[rank][1], "sample order must change between epochs"
        assert both[rank][0] != both[rank][1], "sample order must change between epochs"
    for epoch in range(EPOCHS):
        _assert_partition([both[r][epoch] for r in range(comm.size)], files, f"both shuffles epoch {epoch + 1}")
    clean()
