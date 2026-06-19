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
import math
import pickle
import time
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.data.sampler import Sampler

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DatasetType, DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI, DLIOLogger, Profile, dft_ai
from dlio_benchmark.utils.config import ConfigArguments

dlp = Profile(MODULE_DATA_LOADER)


class TorchDataset(Dataset):
    """
    Currently, we only support loading one sample per file
    TODO: support multiple samples per file
    """

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.num_samples = num_samples
        self.reader = None
        self.num_images_read = 0
        self.batch_size = batch_size
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        self.logger = args.logger
        if num_workers == 0:
            self.worker_init(-1)

    @dlp.log
    def worker_init(self, worker_id):
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        self.logger.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=worker_id,
                                               epoch_number=self.epoch_number)

    @dlp.log
    def __len__(self):
        return self.num_samples

    def __getitem__(self, image_idx):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        self.logger.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {image_idx} sample")
        dlp.update(step=step)
        dft_ai.update(step=step)
        return self.reader.read_index(image_idx, step)


class TorchIterableDataset(IterableDataset):
    """
    Row-Group-granular IterableDataset for high-performance parquet S3 I/O.

    The Map-style TorchDataset calls __getitem__ once per *sample* — 64 million
    times for DLRMv2, burning 212 s in pure Python overhead before any I/O.
    This class iterates at Row-Group granularity instead:

      Python calls = num_files × rgs_per_file / num_workers   (≈ 984 per worker)

    The generator fetches one RG at a time (one S3 GET — GIL released for the
    entire network transfer), accumulates samples, and yields one dummy item per
    complete batch.  Python only resumes at batch boundaries.

    Outer-loop Python cost: 31,250 batch yields × ~3 µs ≈ 0.09 s/epoch.
    vs Map-style:           64,000,000 getitem calls × ~3 µs ≈ 212 s/epoch.

    Only works with readers that expose open()/get_sample() — specifically
    ParquetReaderS3Iterable (storage_library: s3torchconnector / minio / s3dlio).
    """

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, batch_size, num_workers):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.reader = None
        args = ConfigArguments.get_instance()
        opts = getattr(args, "storage_options", {}) or {}
        self._simulate = str(opts.get("simulate_io", "false")).lower() in ("true", "1", "yes")
        self.serial_args = pickle.dumps(args)
        # Only create the reader for in-process (num_workers=0) non-simulate runs.
        # Simulate mode never touches the reader; workers get it via worker_init.
        if num_workers == 0 and not self._simulate:
            self.worker_init(-1)

    @dlp.log
    def worker_init(self, worker_id):
        if self._simulate:
            return  # no reader needed in simulate mode
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        self.reader = ReaderFactory.get_reader(
            type=self.format_type,
            dataset_type=self.dataset_type,
            thread_index=worker_id,
            epoch_number=self.epoch_number,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        args = ConfigArguments.get_instance()
        dummy = args.resized_image
        opts = getattr(args, "storage_options", {}) or {}
        simulate = str(opts.get("simulate_io", "false")).lower() in ("true", "1", "yes")

        if simulate:
            # Dry-run: measure Python loop overhead only — zero I/O, zero reader calls.
            # Total batches distributed evenly across workers.
            total_batches = (args.num_files_train * args.num_samples_per_file) // self.batch_size
            if worker_info is not None:
                n = worker_info.num_workers
                my_batches = total_batches // n + (1 if worker_info.id < total_batches % n else 0)
            else:
                my_batches = total_batches
            wid = worker_info.id if worker_info is not None else 0
            print(f"[ITER] worker={wid} path=SIMULATE total_batches={total_batches} my_batches={my_batches}", flush=True)
            for _ in range(my_batches):
                yield dummy
            return

        # _file_list is set by FormatReader.__init__ from args.file_list_train
        all_files = list(self.reader._file_list)

        # Shard files across workers: worker k handles files[k::num_workers]
        if worker_info is not None:
            all_files = all_files[worker_info.id::worker_info.num_workers]

        wid = worker_info.id if worker_info is not None else 0
        reader_type = type(self.reader).__name__
        print(f"[ITER] worker={wid} reader={reader_type} files_this_worker={len(all_files)} total_files={len(list(self.reader._file_list))}", flush=True)

        # ── s3dlio consumer-driven pipeline ─────────────────────────────────
        # ParquetReaderS3dlio.iter_epoch() installs the worker's file shard,
        # runs _epoch_init() on only ~N/W files, and drives the bounded
        # sliding-window pipeline directly — no pyarrow 3-tuple unpacking.
        if hasattr(self.reader, 'iter_epoch'):
            print(f"[ITER] worker={wid} path=ITER_EPOCH files={len(all_files)}", flush=True)
            yield from self.reader.iter_epoch(all_files, self.batch_size)
            return

        # ── Sliding-window prefetch ─────────────────────────────────────────
        # Instead of firing all RG GETs for a file at once (burst → idle →
        # burst), we maintain a constant window of `prefetch_window` in-flight
        # GETs drawn from a flat queue of (filename, rg_idx) tuples spanning
        # ALL files.  As each RG is consumed one new slot is filled, keeping
        # the network continuously saturated.
        #
        # Use the reader's sliding-window helpers when available
        # (ParquetReaderS3Iterable); fall back to the old open()-everything
        # path for other readers.
        has_sliding = (
            hasattr(self.reader, 'open_footer_only')
            and hasattr(self.reader, 'submit_rg_prefetch')
            and self.reader._prefetch_executor is not None
        )

        if not has_sliding:
            # ── Legacy path (no sliding window) ────────────────────────────
            sample_buf = 0
            file_iter = iter(all_files)
            next_filename = next(file_iter, None)
            if next_filename is None:
                return
            next_data = self.reader.open(next_filename)
            self.reader.open_file_map[next_filename] = next_data
            while next_filename is not None:
                filename = next_filename
                file_data = next_data
                next_filename = next(file_iter, None)
                if next_filename is not None:
                    next_data = self.reader.open(next_filename)
                    self.reader.open_file_map[next_filename] = next_data
                else:
                    next_data = None
                pf, rf, offsets = file_data
                num_rgs = pf.metadata.num_row_groups
                for rg_idx in range(num_rgs):
                    self.reader.get_sample(filename, offsets[rg_idx])
                    rg_rows = offsets[rg_idx + 1] - offsets[rg_idx]
                    sample_buf += rg_rows
                    while sample_buf >= self.batch_size:
                        yield dummy
                        sample_buf -= self.batch_size
                if hasattr(self.reader, '_pf_cache'):
                    self.reader._pf_cache.pop(filename, None)
                self.reader.open_file_map[filename] = None
            return

        # ── Sliding-window path ─────────────────────────────────────────────
        # Step 1: fetch all footers (cheap: one small range GET per file, can
        #         run serially — footer size is ~50 KB vs ~8 MB per RG).
        file_meta = {}   # filename -> (pf, rf, offsets)
        for fn in all_files:
            fd = self.reader.open_footer_only(fn)
            self.reader.open_file_map[fn] = fd
            file_meta[fn] = fd

        # Step 2: build flat RG queue across all files in order.
        rg_queue = []   # list of (filename, rg_idx, rg_rows)
        for fn in all_files:
            pf, rf, offsets = file_meta[fn]
            num_rgs = pf.metadata.num_row_groups
            for rg_idx in range(num_rgs):
                rg_rows = offsets[rg_idx + 1] - offsets[rg_idx]
                rg_queue.append((fn, rg_idx, rg_rows))

        # Step 3: fill the initial window — submit first `window` GETs.
        # Window size: default 64, overridable via storage_options.prefetch_window.
        opts = getattr(args, "storage_options", {}) or {}
        window = int(opts.get("prefetch_window", 64))
        tail = 0  # index of next RG to submit into the window

        # t_io_start: wall-clock time when the very first GET is submitted.
        # This excludes DLIO startup, footer fetches, and queue construction.
        t_io_start = time.perf_counter()
        for tail in range(min(window, len(rg_queue))):
            fn, rg_idx, _ = rg_queue[tail]
            self.reader.submit_rg_prefetch(fn, rg_idx)
        tail = min(window, len(rg_queue))

        # Step 4: consume RGs in order, refilling the window by 1 per RG consumed.
        sample_buf = 0
        prev_filename = None
        total_bytes = 0  # compressed bytes fetched by this worker
        for consume_idx, (filename, rg_idx, rg_rows) in enumerate(rg_queue):
            # Refill: submit the RG that is `window` ahead of the one we're consuming.
            if tail < len(rg_queue):
                fn_ahead, rg_ahead, _ = rg_queue[tail]
                self.reader.submit_rg_prefetch(fn_ahead, rg_ahead)
                tail += 1

            # Wait for this RG (already in-flight; should be near-instant).
            self.reader.get_sample(filename, file_meta[filename][2][rg_idx])
            # Accumulate bytes from the rg_cache (populated by get_sample).
            if hasattr(self.reader, '_rg_cache'):
                total_bytes += self.reader._rg_cache.get((filename, rg_idx), 0)
            sample_buf += rg_rows

            # Release the previous file's caches once we move on to a new file.
            if prev_filename is not None and prev_filename != filename:
                if hasattr(self.reader, '_pf_cache'):
                    self.reader._pf_cache.pop(prev_filename, None)
                if hasattr(self.reader, '_rg_cache'):
                    stale = [k for k in list(self.reader._rg_cache) if k[0] == prev_filename]
                    for k in stale:
                        self.reader._rg_cache.pop(k, None)
                self.reader.open_file_map[prev_filename] = None
            prev_filename = filename

            while sample_buf >= self.batch_size:
                yield dummy
                sample_buf -= self.batch_size

        # t_io_end: wall-clock time after the last RG has been waited on.
        t_io_end = time.perf_counter()

        # Clean up last file.
        if prev_filename is not None:
            if hasattr(self.reader, '_pf_cache'):
                self.reader._pf_cache.pop(prev_filename, None)
            self.reader.open_file_map[prev_filename] = None

        # Report per-worker I/O throughput (excludes startup, footer fetches,
        # and any time spent in the PyTorch training loop between yields).
        elapsed = t_io_end - t_io_start
        gib = total_bytes / (1024 ** 3)
        mib_s = (total_bytes / (1024 ** 2)) / elapsed if elapsed > 0 else 0.0
        wid = worker_info.id if worker_info is not None else 0
        print(
            f"[io_timing] worker={wid} "
            f"bytes={total_bytes} ({gib:.3f} GiB) "
            f"elapsed={elapsed:.3f}s "
            f"throughput={mib_s:.1f} MiB/s",
            flush=True,
        )


class TorchIterableDatasetSimple(IterableDataset):
    """
    IterableDataset for 1-sample-per-file formats (NPZ / NPY / JPEG / PNG) on
    any storage backend (S3 object store or local / POSIX filesystem).

    Problem with map-style (TorchDataset) for these formats:
      Each worker calls __getitem__ → read_index() → on-demand single-object GET.
      With N workers the server sees at most N simultaneous requests.

    This class instead calls reader.next(), which:
      S3   — _s3_prefetch_all()      → s3dlio.get_many(64 in-flight per worker)
      POSIX — _localfs_prefetch_all() → ThreadPoolExecutor(64 workers)

    Effective pipeline depth: 64 × num_workers  (vs 1 × num_workers before).

    File assignment:
      Worker k handles files[k::num_workers] from the epoch's file list.
      The list reflects any epoch-level shuffle already applied by reconfigure().
      No additional shuffle is performed here; for storage I/O benchmarking,
      file ordering within a worker's shard does not affect measured throughput.

    Drop-last semantics:
      FormatReader.next() drops the final partial batch (same as map-style
      drop_last=True). The DataLoader is configured with batch_size=None because
      the reader assembles batches internally; one 'dummy' item is yielded per
      complete batch.
    """

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, batch_size, num_workers):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.reader = None
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        if num_workers == 0:
            self.worker_init(-1)

    @dlp.log
    def worker_init(self, worker_id):
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        # When num_workers=0 the training loop runs in the main process;
        # file_map is built for thread 0 (num_threads=1 when read_threads=0).
        thread_idx = 0 if worker_id < 0 else worker_id
        self.reader = ReaderFactory.get_reader(
            type=self.format_type,
            dataset_type=self.dataset_type,
            thread_index=thread_idx,
            epoch_number=self.epoch_number,
        )

    def __iter__(self):
        args = ConfigArguments.get_instance()
        dummy = args.resized_image
        worker_info = torch.utils.data.get_worker_info()

        # Shard files: worker k handles every k-th file starting at offset k.
        # _file_list = args.file_list_train, updated each epoch by reconfigure().
        all_files = list(self.reader._file_list)
        if worker_info is not None:
            my_files = all_files[worker_info.id::worker_info.num_workers]
        else:
            my_files = all_files

        wid = worker_info.id if worker_info is not None else 0
        reader_type = type(self.reader).__name__
        print(
            f"[ITER_SIMPLE] worker={wid} reader={reader_type} "
            f"files_this_worker={len(my_files)} total_files={len(all_files)}",
            flush=True,
        )

        if not my_files:
            return

        # Install the file shard into the reader's file_map so reader.next()
        # picks it up. Entry format: (global_sample_idx, filename, sample_in_file).
        num_spf = args.num_samples_per_file
        entries = [
            (i * num_spf + s, filename, s)
            for i, filename in enumerate(my_files)
            for s in range(num_spf)
        ]
        self.reader.file_map[self.reader.thread_index] = entries

        # reader.next() bulk-prefetches all files in this worker's shard, then
        # iterates them yielding one numpy batch per batch_size samples.
        # We yield one dummy item per batch so the DataLoader step count is correct.
        for _batch in self.reader.next():
            yield dummy

    @dlp.log
    def finalize(self):
        if self.reader is not None:
            self.reader.finalize()


class dlio_sampler(Sampler):
    def __init__(self, rank, size, num_samples, epochs):
        self.size = size
        self.rank = rank
        self.num_samples = num_samples
        self.epochs = epochs
        try:
            pre_sharded = ConfigArguments.get_instance().files_pre_sharded
        except Exception:
            # MPI not initialized (e.g. unit tests) — treat as non-pre-sharded.
            pre_sharded = False
        if pre_sharded:
            # Files already distributed — this rank owns all its local samples.
            # Round-robin gives even counts, but allreduce_min as safety net.
            aligned = DLIOMPI.get_instance().allreduce_min(num_samples)
            self.indices = list(range(aligned))
            self.num_samples = aligned
        else:
            # Use floor division so every rank gets the same sample count. With
            # math.ceil() the last rank was clamped to fewer samples than its
            # peers when num_samples % size != 0; mismatched per-rank batch
            # counts caused the per-step and end-of-epoch barriers in main._train()
            # to match across iterations and deadlock at the next epoch boundary.
            # The trailing (num_samples % size) samples are dropped on purpose;
            # pick num_samples as a multiple of comm_size to use every sample.
            samples_per_proc = num_samples // size
            start_sample = self.rank * samples_per_proc
            end_sample = (self.rank + 1) * samples_per_proc - 1
            self.indices = list(range(start_sample, end_sample + 1))
            dropped = num_samples - samples_per_proc * size
            if dropped > 0 and self.rank == 0:
                DLIOLogger.get_instance().warning(
                    f"{utcnow()} dlio_sampler: dropping {dropped} sample(s) — "
                    f"num_samples ({num_samples}) is not a multiple of comm_size "
                    f"({size}). Each rank will process {samples_per_proc} samples. "
                    f"Choose num_samples as a multiple of {size} to use every sample."
                )


    def __len__(self):
        # Per-rank shard length — must match what __iter__ yields. Returning
        # self.num_samples (the global count) here is a pre-existing bug that
        # the floor-division change above makes provable: len(self.indices) is
        # now num_samples // size while self.num_samples is still num_samples,
        # so any caller that builds len(DataLoader) from len(sampler) would
        # over-report by a factor of comm_size.
        return len(self.indices)

    def __iter__(self):
        for sample in self.indices:
            yield sample


class TorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.PYTORCH)

    @dlp.log
    def read(self):
        from dlio_benchmark.common.enumerations import FormatType, StorageType
        from dlio_benchmark.reader.reader_factory import ReaderFactory

        # ── Dataset routing ──────────────────────────────────────────────────
        # Three paths, in priority order:
        #
        # 1. PARQUET on any storage → TorchIterableDataset (RG-granular)
        #    Reduces Python __getitem__ calls from O(samples) to O(row_groups).
        #
        # 2. NPZ / NPY / JPEG / PNG → TorchIterableDatasetSimple (bulk-prefetch)
        #    reader.next() calls _s3_prefetch_all() or _localfs_prefetch_all()
        #    before iteration, giving 64 × num_workers in-flight I/O requests
        #    instead of 1 × num_workers with map-style on-demand fetching.
        #    Works for BOTH object storage and POSIX/local filesystem.
        #
        # 3. Everything else → TorchDataset (map-style, on-demand per sample)

        _simple_iterable_formats = (
            FormatType.NPZ, FormatType.NPY, FormatType.JPEG, FormatType.PNG
        )
        use_rg_iterable_dataset = (
            self.format_type is FormatType.PARQUET
            and self._args.storage_type in (
                StorageType.S3, StorageType.AISTORE,
                StorageType.LOCAL_FS,
            )
        )
        _s3_types = (StorageType.S3, StorageType.AISTORE)
        # TorchIterableDatasetSimple uses DataLoader(num_workers>0) which forks
        # worker processes via os.fork(). On LOCAL_FS, this fork-after-module-import
        # pattern causes a ThreadPoolExecutor deadlock (the executor's background
        # thread is not fork-safe). Restrict the iterable path to object storage
        # (S3/AISTORE) only where the prefetch benefit is most significant and
        # the fork issue does not apply. LOCAL_FS falls through to map-style TorchDataset.
        use_simple_iterable_dataset = (
            self.format_type in _simple_iterable_formats
            and not use_rg_iterable_dataset
            and self._args.storage_type in _s3_types
        )

        # Determine concrete reader class name and access pattern for logging.
        _opts = getattr(self._args, "storage_options", {}) or {}
        _lib  = _opts.get("storage_library", "none")
        _st   = self._args.storage_type
        _s3_libs  = ("s3dlio", "s3torchconnector", "minio")
        _nw = self._args.read_threads

        if use_rg_iterable_dataset:
            _reader_cls    = "ParquetReaderS3dlio"
            _torch_ds      = "TorchIterableDataset(rg-granular)"
            _sample_access = "iterator(row-group chunks)"
        elif use_simple_iterable_dataset:
            _fmt = str(self.format_type).split(".")[-1].lower()
            if _st in _s3_types and _lib in _s3_libs:
                _reader_cls = (
                    "NPZReaderS3Iterable"   if self.format_type is FormatType.NPZ  else
                    "NPYReaderS3Iterable"   if self.format_type is FormatType.NPY  else
                    "ImageReaderS3Iterable"
                )
                _sample_access = "next()→_s3_prefetch_all(64 in-flight) then iterate"
            else:
                _reader_cls = (
                    "NPZReaderIterable"    if self.format_type is FormatType.NPZ  else
                    "NPYReaderIterable"    if self.format_type is FormatType.NPY  else
                    "ImageReaderIterable"
                )
                _sample_access = "next()→_localfs_prefetch_all(ThreadPool-64) then iterate"
            _torch_ds = f"TorchIterableDatasetSimple(bulk-prefetch, {_nw} workers)"
        else:
            _reader_cls    = "unknown"
            _torch_ds      = f"TorchDataset(map-style, {_nw} workers)"
            _sample_access = "read_index (on-demand)"

        print(
            f"[DATALOADER] format={self.format_type} storage={_st} library={_lib}\n"
            f"[DATALOADER]   torch_dataset={_torch_ds}\n"
            f"[DATALOADER]   reader={_reader_cls}\n"
            f"[DATALOADER]   sample_access={_sample_access}",
            flush=True,
        )

        if use_simple_iterable_dataset:
            num_workers = self._args.read_threads
            dataset = TorchIterableDatasetSimple(
                self.format_type, self.dataset_type, self.epoch_number,
                self.batch_size, num_workers,
            )
            if self._args.my_rank == 0:
                self.logger.debug(
                    f"{utcnow()} Using TorchIterableDatasetSimple: "
                    f"{num_workers} workers, batch_size={self.batch_size}, "
                    f"reader={_reader_cls}"
                )
            if num_workers == 0:
                kwargs = {}
            else:
                kwargs = {'multiprocessing_context': self._args.multiprocessing_context}
                if torch.__version__ != '1.3.1':
                    kwargs['persistent_workers'] = True
            pin_memory = self._args.pin_memory and torch.cuda.is_available()
            self._dataset = DataLoader(
                dataset,
                batch_size=None,          # reader assembles batches; one dummy per batch
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=dataset.worker_init,
                **kwargs,
            )
            return

        if use_rg_iterable_dataset:
            opts = getattr(self._args, "storage_options", {}) or {}
            simulate = str(opts.get("simulate_io", "false")).lower() in ("true", "1", "yes")
            # For simulate: run single-process — no multiprocessing IPC overhead.
            # s3torchconnector/CRT releases the GIL during network I/O, so a
            # thread-pool in the reader provides concurrency without IPC cost.
            num_workers = 0 if simulate else self._args.read_threads
            dataset = TorchIterableDataset(
                self.format_type, self.dataset_type, self.epoch_number,
                self.batch_size, num_workers,
            )
            if self._args.my_rank == 0:
                mode = "simulate/single-process" if simulate else f"{num_workers} workers"
                self.logger.debug(
                    f"{utcnow()} Using TorchIterableDataset (RG-granular): "
                    f"{mode}, batch_size={self.batch_size}"
                )
            if num_workers == 0:
                kwargs = {}
            else:
                kwargs = {'multiprocessing_context': self._args.multiprocessing_context}
                if torch.__version__ != '1.3.1':
                    kwargs['persistent_workers'] = True
            pin_memory = self._args.pin_memory and torch.cuda.is_available()
            self._dataset = DataLoader(
                dataset,
                batch_size=None,          # __iter__ yields pre-formed batch items
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=dataset.worker_init,
                **kwargs,
            )
            return

        dataset = TorchDataset(self.format_type, self.dataset_type, self.epoch_number, self.num_samples,
                               self._args.read_threads, self.batch_size)
        sampler = dlio_sampler(self._args.my_rank, self._args.comm_size, self.num_samples, self._args.epochs)
        if self._args.read_threads >= 1:
            prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
        else:
            prefetch_factor = self._args.prefetch_size
        if prefetch_factor > 0:
            if self._args.my_rank == 0:
                self.logger.debug(
                    f"{utcnow()} Prefetch size is {self._args.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            prefetch_factor = 2
            if self._args.my_rank == 0:
                self.logger.debug(
                    f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
        self.logger.debug(f"{utcnow()} Setup dataloader with {self._args.read_threads} workers {torch.__version__}")
        if self._args.read_threads==0:
            kwargs={}
        else:
            kwargs={'multiprocessing_context':self._args.multiprocessing_context,
                    'prefetch_factor': prefetch_factor}
            # persistent_workers=False: workers re-spawn each epoch to pick up
            # resharded file lists from updated serial_args.
        pin_memory = self._args.pin_memory and torch.cuda.is_available()
        if torch.__version__ == '1.3.1':
            if 'prefetch_factor' in kwargs:
                del kwargs['prefetch_factor']
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=pin_memory,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init, 
                                       **kwargs)
        else: 
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=pin_memory,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init,
                                       **kwargs)  # 2 is the default value
        self.logger.debug(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataset) * self.batch_size} files")

        # self._dataset.sampler.set_epoch(epoch_number)

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        self.logger.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        step = 1
        for batch in dft_ai.dataloader.fetch.iter(self._dataset):
            dlp.update(step=step)
            dft_ai.update(step=step)
            step += 1
            yield batch
        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)
        dft_ai.update(epoch=self.epoch_number)

    def refresh_args(self):
        """Re-serialize ConfigArguments so next worker spawn picks up changes.

        Call after reconfigure() to propagate resharded file lists to workers.
        Without persistent_workers, workers re-spawn on each iter() and call
        worker_init → pickle.loads(serial_args), so this ensures freshness.
        """
        dataset = self._dataset.dataset
        if hasattr(dataset, 'serial_args'):
            dataset.serial_args = pickle.dumps(self._args)

    @dlp.log
    def finalize(self):
        # When read_threads=0 the reader lives in-process on the dataset object.
        # Call its finalize() so per-epoch state (byte counters, etc.) is flushed
        # back to ConfigArguments before statscounter computes the I/O metric.
        # When read_threads>0 each worker is a separate process; their readers are
        # not accessible here.  The workers call finalize_s3_bytes() internally
        # (logging actual bytes), but cannot update the main-process args.record_length.
        # The config.py fix (guarding record_dims overwrite) ensures the correct
        # record_length_bytes value from the YAML is used in that case.
        try:
            dataset = self._dataset.dataset
            # TorchIterableDatasetSimple exposes finalize() directly.
            if isinstance(dataset, TorchIterableDatasetSimple):
                dataset.finalize()
            elif dataset.reader is not None:
                dataset.reader.finalize()
        except Exception:
            pass
