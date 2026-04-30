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

from abc import ABC, abstractmethod
import io
from concurrent.futures import ThreadPoolExecutor

from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.storage.storage_factory import StorageFactory
import numpy as np
from dlio_benchmark.utils.utility import utcnow, add_padding, DLIOMPI, Profile, progress
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp_base = Profile(MODULE_DATA_GENERATOR)


class DataGenerator(ABC):

    # Fixed base seed shared by all generators.
    # Per-file seed = BASE_SEED + global_file_index, which gives each file a
    # unique-but-reproducible seed: identical across runs, different per rank.
    BASE_SEED: int = 10

    def __init__(self):
        self._args = ConfigArguments.get_instance()
        # Issue 6b note: derive_configurations() here runs the early (no file list) path.
        # validate() is NOT called here — it is called in main.py after the file list walk.
        # This is intentional: validate() checks file counts which aren't known until walk.
        self._args.derive_configurations()
        self._dimension = self._args.dimension
        self._dimension_stdev = self._args.dimension_stdev
        self.data_dir = self._args.data_folder
        self.file_prefix = self._args.file_prefix
        self.num_files_train = self._args.num_files_train
        self.do_eval = self._args.do_eval
        self.num_files_eval = self._args.num_files_eval
        self.num_samples = self._args.num_samples_per_file
        self.my_rank = self._args.my_rank
        self.comm_size = self._args.comm_size
        self.compression = self._args.compression
        self.compression_level = self._args.compression_level
        self._file_prefix = None
        self._file_list = None
        self.num_subfolders_train = self._args.num_subfolders_train
        self.num_subfolders_eval = self._args.num_subfolders_eval
        self.format = self._args.format
        self.logger = self._args.logger
        self.storage = StorageFactory().get_storage(self._args.storage_type, self._args.storage_root,
                                                                        self._args.framework)

    def _file_seed(self, i: int) -> int:
        """Return the reproducible per-file seed for global file index *i*.

        Properties:
        - **Reproducible**: ``BASE_SEED + i`` is a pure function of fixed values,
          so the same file index always produces the same seed across runs.
        - **Unique per file**: ``i`` uniquely identifies each file across all MPI
          ranks (rank *r* processes files where ``i % comm_size == r``), so no two
          files ever share a seed.
        - **Unique per rank**: since ``i % comm_size`` differs per rank, files
          processed by different ranks have disjoint seed ranges.
        """
        return self.BASE_SEED + i

    @staticmethod
    def _extract_dims(dim, i):
        """Extract scalar dimensions from the dimension array at position *i*.

        Returns ``(dim_raw, dim1, dim2)`` where:
        - ``dim_raw``: the raw value from ``dim[2*i]`` (list or int)
        - ``dim1``: first scalar dimension (int)
        - ``dim2``: second scalar dimension (int; 1 when ``dim_raw`` is a
          single-element list)
        """
        dim_raw = dim[2 * i]
        if isinstance(dim_raw, list):
            dim1 = int(dim_raw[0])
            dim2 = int(dim_raw[1]) if len(dim_raw) > 1 else 1
        else:
            dim1 = int(dim_raw)
            dim2 = int(dim[2 * i + 1])
        return dim_raw, dim1, dim2

    def _generate_files(self, write_fn, label: str = "Data") -> None:
        """Template for the standard per-file generation loop.

        Handles:
        - Rank-unique, reproducible numpy global seed for ``get_dimension()``.
        - Dimension extraction (scalar / list branch).
        - BytesIO abstraction for object storage.
        - ``storage.put_data()`` after each file when not on local FS.
        - Parallel file writes via ``ThreadPoolExecutor`` when ``write_threads > 1``.

        **write_fn signature**::

            write_fn(i, dim_, dim1, dim2, file_seed, rng,
                     out_path_spec, is_local, output) -> None

        Parameters passed to write_fn:

        - ``i``            : global file index (unique per file across all ranks)
        - ``dim_``         : raw dimension from ``get_dimension()`` (list or int)
        - ``dim1, dim2``   : extracted scalar first/second dimensions
        - ``file_seed``    : reproducible per-file seed (int64).  Each worker
                             creates its own ``np.random.default_rng(file_seed)``
                             so that: (a) no shared mutable state crosses thread
                             boundaries, and (b) the same config always generates
                             identical files regardless of ``write_threads``.
        - ``rng``          : ``np.random.Generator`` seeded from ``file_seed`` —
                             a fresh instance per file, safe for concurrent use.
        - ``out_path_spec``: fully-resolved path string
        - ``is_local``     : ``True`` for local filesystem, ``False`` for object store
        - ``output``       : ``out_path_spec`` when ``is_local``,
                             ``io.BytesIO()`` when not

        After ``write_fn`` returns, if ``not is_local``, the template calls::

            storage.put_data(out_path_spec, output.getvalue())

        **Parallel semantics** (Issue 10):

        Seeds are pre-derived sequentially in the main thread so that
        determinism is preserved: ``same master seed → same per-file seeds →
        identical files`` regardless of ``write_threads`` value.
        Worker threads each receive a pre-computed seed and create their own
        independent ``np.random.Generator`` — no shared RNG state.
        """
        # Rank-unique seed for get_dimension() global random state.
        np.random.seed(self.BASE_SEED + self.my_rank)
        rng = np.random.default_rng(seed=self.BASE_SEED + self.my_rank)
        dim = self.get_dimension(self.total_files_to_generate)
        is_local = self.storage.islocalfs()

        # Phase 1: Pre-derive all (index, dims, seed, path) in the main thread.
        # rng.integers() calls MUST happen in order to preserve the deterministic
        # sequence; workers receive pre-computed seeds and never touch this rng.
        jobs = []
        for i in dlp_base.iter(range(self.my_rank,
                                     int(self.total_files_to_generate),
                                     self.comm_size)):
            dim_, dim1, dim2 = self._extract_dims(dim, i)
            out_path_spec = self.storage.get_uri(self._file_list[i])
            file_seed = int(rng.integers(0, 2**63))
            jobs.append((i, dim_, dim1, dim2, file_seed, out_path_spec))

        # Phase 2: Execute writes, optionally in parallel.
        # Each worker creates a fresh rng from its pre-derived file_seed so
        # there is no shared mutable state between threads.
        def _write_one(job):
            i, dim_, dim1, dim2, file_seed, out_path_spec = job
            progress(i + 1, self.total_files_to_generate, f"Generating {label}")
            output = out_path_spec if is_local else io.BytesIO()
            worker_rng = np.random.default_rng(seed=file_seed)
            write_fn(i, dim_, dim1, dim2, file_seed, worker_rng,
                     out_path_spec, is_local, output)
            if not is_local:
                # Pass BytesIO directly so put_data can use getbuffer() (zero-copy
                # memoryview) instead of getvalue() which makes a full copy.
                self.storage.put_data(out_path_spec, output)

        write_threads = getattr(self._args, 'write_threads', 1)
        n_workers = max(1, min(write_threads, len(jobs))) if jobs else 1

        # Phase 2: Execute writes.
        #
        # Two strategies depending on storage type:
        #
        # A) Object store (not is_local) with n_workers > 1 — TRUE ASYNC PIPELINE:
        #    Generation is very fast (dgen-py / Rust, ~15 GB/s), while each
        #    upload takes ~500 ms–1.5 s (limited by network/server throughput).
        #    Coupling generation and upload in the same thread wastes the time
        #    the uploader is blocked waiting for network.
        #
        #    Instead:
        #      • Main thread generates files sequentially (~14 ms each).
        #      • Each file is immediately submitted to an upload thread pool.
        #      • A bounded semaphore (size = n_workers) prevents submitting more
        #        than n_workers uploads before any have completed, bounding
        #        peak in-flight memory to n_workers × file_size.
        #
        #    With n_workers=8 (NP=8 on a 28-core machine):
        #      Generation of all files: 21 files × 14 ms = ~300 ms
        #      8 concurrent uploads running throughout → ~4× improvement
        #      vs. the old 3-thread combined gen+upload approach.
        #
        # B) Local FS, or single-threaded: original thread pool approach.
        #    For local FS, write is CPU-bound and the combined thread pool
        #    (generate + write in each slot) is correct.
        def _upload(path, buf, sem):
            """Upload helper: calls put_data then releases the semaphore slot."""
            try:
                self.storage.put_data(path, buf)
            finally:
                sem.release()

        if not is_local and n_workers > 1:
            # ── True async pipeline (object store) ───────────────────────────
            from threading import Semaphore as _Semaphore
            _sem = _Semaphore(n_workers)
            _futures = []
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for job in jobs:
                    i, dim_, dim1, dim2, file_seed, out_path_spec = job
                    progress(i + 1, self.total_files_to_generate,
                             f"Generating {label}")
                    output = io.BytesIO()
                    worker_rng = np.random.default_rng(seed=file_seed)
                    # Generate in main thread (fast; Rust dgen or numpy)
                    write_fn(i, dim_, dim1, dim2, file_seed, worker_rng,
                             out_path_spec, False, output)
                    # Block if n_workers uploads are already in flight
                    # (back-pressure to bound peak RAM usage).
                    _sem.acquire()
                    # Submit upload immediately; main thread continues generating.
                    _futures.append(
                        pool.submit(_upload, out_path_spec, output, _sem)
                    )
                # Wait for all in-flight uploads before leaving the with block.
                for f in _futures:
                    f.result()

        elif n_workers == 1 or len(jobs) <= 1:
            # ── Serial fallback ───────────────────────────────────────────────
            for job in jobs:
                _write_one(job)
        else:
            # ── Local FS thread pool (generate + write per slot) ──────────────
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(_write_one, jobs))

        np.random.seed()  # Reset global seed to avoid leaking state

    def get_dimension(self, num_samples=1):
        if isinstance(self._dimension, list):
            if self._dimension_stdev > 0:
                # Generated shape (2*num_samples, len(self._dimension))
                random_values = np.random.normal(
                    loc=self._dimension,
                    scale=self._dimension_stdev,
                    size=(2 * num_samples, len(self._dimension))
                )
                dim = np.maximum(random_values.astype(int), 1).tolist()
            else:
                dim = [self._dimension for _ in range(2 * num_samples)]

            return dim

        if (self._dimension_stdev>0):
            dim = [max(int(d), 1) for d in np.random.normal(self._dimension, self._dimension_stdev, 2*num_samples)]
        else:
            dim = np.ones(2*num_samples, dtype=np.int64)*int(self._dimension)
        return dim 

    @abstractmethod
    def generate(self):
        nd_f_train = len(str(self.num_files_train))
        nd_f_eval = len(str(self.num_files_eval))
        nd_sf_train = len(str(self.num_subfolders_train))
        nd_sf_eval = len(str(self.num_subfolders_eval))

        if self.my_rank == 0:
            self.storage.create_node(self.data_dir, exist_ok=True)
            self.storage.create_node(self.data_dir + "/train/", exist_ok=True)
            self.storage.create_node(self.data_dir + "/valid/", exist_ok=True)
            if self.num_subfolders_train > 1: 
                for i in range(self.num_subfolders_train):
                    self.storage.create_node(self.data_dir + f"/train/{add_padding(i, nd_sf_train)}", exist_ok=True)
            if self.num_subfolders_eval > 1: 
                for i in range(self.num_subfolders_eval):
                    self.storage.create_node(self.data_dir + f"/valid/{add_padding(i, nd_sf_eval)}", exist_ok=True)
            self.logger.info(f"{utcnow()} Generating dataset in {self.data_dir}/train and {self.data_dir}/valid")
            self.logger.info(f"{utcnow()} Number of files for training dataset: {self.num_files_train}")
            self.logger.info(f"{utcnow()} Number of files for validation dataset: {self.num_files_eval}")


        DLIOMPI.get_instance().comm().barrier()
        # What is the logic behind this formula? 
        # Will probably have to adapt to generate non-images
        self.total_files_to_generate = self.num_files_train
        if self.num_files_eval > 0:
            self.total_files_to_generate += self.num_files_eval
        self._file_list = []


        if self.num_subfolders_train > 1:
            ns = np.ceil(self.num_files_train / self.num_subfolders_train)
            for i in range(self.num_files_train):
                file_spec = "{}/train/{}/{}_{}_of_{}.{}".format(self.data_dir, add_padding(i%self.num_subfolders_train, nd_sf_train), self.file_prefix, add_padding(i, nd_f_train), self.num_files_train, self.format)
                self._file_list.append(file_spec)
        else:
            for i in range(self.num_files_train):
                file_spec = "{}/train/{}_{}_of_{}.{}".format(self.data_dir, self.file_prefix, add_padding(i, nd_f_train), self.num_files_train, self.format)
                self._file_list.append(file_spec)
        if self.num_subfolders_eval > 1:
            ns = np.ceil(self.num_files_eval / self.num_subfolders_eval)
            for i in range(self.num_files_eval):
                file_spec = "{}/valid/{}/{}_{}_of_{}.{}".format(self.data_dir, add_padding(i%self.num_subfolders_eval, nd_sf_eval), self.file_prefix, add_padding(i, nd_f_eval), self.num_files_eval, self.format)
                self._file_list.append(file_spec)
        else:
            for i in range(self.num_files_eval):
                file_spec = "{}/valid/{}_{}_of_{}.{}".format(self.data_dir, self.file_prefix, add_padding(i, nd_f_eval), self.num_files_eval, self.format)
                self._file_list.append(file_spec)
