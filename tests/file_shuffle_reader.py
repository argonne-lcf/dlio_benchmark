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

Read-recording reader used by tests/dlio_file_shuffle_test.py.

It is plugged in through ``workload.reader.reader_classname`` and behaves exactly
like IndexedBinaryReader, except that every sample read is also logged together
with the byte offset and length that went through the data file handle. The
tests use the log to check the order in which files and samples were really
read on each rank.

The module imports no MPI on purpose: DataLoader workers inherit it (fork) or
import it (spawn, forkserver).

Records are written to ``$DLIO_TEST_READ_LOG/reads.<pid>.tsv``, one line per
sample read, with the columns of READ_LOG_COLUMNS. Nothing is logged when the
variable is not set.
"""
import os
import time

from dlio_benchmark.reader.indexed_binary_reader import IndexedBinaryReader

READ_LOG_ENV = "DLIO_TEST_READ_LOG"
READ_LOG_COLUMNS = ("rank", "dataset_type", "pid", "seq", "t_ns", "path", "sample_index", "offset", "nbytes")


class _RecordingFile:
    """Wraps a data file handle and remembers where the last readinto() read from."""

    def __init__(self, f):
        self._f = f
        self.last_read = None

    def readinto(self, buffer):
        offset = self._f.tell()
        nbytes = self._f.readinto(buffer)
        self.last_read = (offset, nbytes)
        return nbytes

    def __getattr__(self, name):
        return getattr(self._f, name)


class _ReadLog:
    """One append-only log file per process and log directory."""
    key = None
    out = None
    seq = 0

    @classmethod
    def write(cls, fields):
        log_dir = os.environ.get(READ_LOG_ENV)
        if not log_dir:
            return
        key = (os.getpid(), log_dir)
        if cls.key != key:
            # first write in this process (or a forked child) or for a new run
            cls.key, cls.seq = key, 0
            cls.out = open(os.path.join(log_dir, f"reads.{os.getpid()}.tsv"), "a", buffering=1)
        cls.seq += 1
        cls.out.write("\t".join(str(v) for v in fields(cls.seq)) + "\n")


class ReadRecordingIndexedBinaryReader(IndexedBinaryReader):

    def open(self, filename):
        return _RecordingFile(super().open(filename))

    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        offset, nbytes = self.open_file_map[filename].last_read
        _ReadLog.write(lambda seq: (self._args.my_rank, self.dataset_type, os.getpid(), seq,
                                    time.monotonic_ns(), filename, sample_index, offset, nbytes))
