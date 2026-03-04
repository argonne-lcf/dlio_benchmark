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
import numpy as np
import pyarrow.parquet as pq
import time
import logging
import sys

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import Profile, utcnow
from dlio_benchmark.reader.reader_handler import FormatReader

dlp = Profile(MODULE_DATA_READER)

# Configure module-level logger to ensure visibility
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

# Add stderr handler if not already present to ensure logs are visible
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    _logger.addHandler(_handler)


class ParquetReader(FormatReader):
    """
    Optimized Parquet reader for file-shuffle-only workloads (sequential sample access).

    This implementation is optimized for I/O benchmarking with file shuffle where
    samples within each file are accessed sequentially. Key optimizations:
    
    1. Single row-group cache: Caches the current row group since sequential access
       means consecutive samples are in the same row group.
    
    2. Zero-copy reads: Uses PyArrow's memory-mapped I/O without converting to
       pandas/numpy, avoiding 2+ memory copies.
    
    3. Lazy row-group loading: Only reads the row group containing the requested
       sample, not the entire file.
    
    4. Simple row group lookup: Uses division for O(1) row group identification
       (assumes uniform row group sizes, which is typical for generated data).

    Supports two read modes:
    - "default": Lazy loading with on-demand row group reads.
    - "row_group": Same as default (both use row-group-based access).

    When parquet_columns is configured, only those columns are read from disk.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.parquet_file = None
        self.field_specs = getattr(self._args, 'parquet_field_specs', {})
        
        # Single row-group cache for sequential access optimization
        self._row_group_cache = None
        self._cache_row_group_idx = -1
        self._cache_filename = None
        
        # Throughput tracking for diagnostics
        self._sample_count = 0
        self._last_log_time = time.time()
        self._log_interval_seconds = 5.0  # Log every 5 seconds
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_samples_processed = 0
        self._logger = _logger  # Use module-level logger
        
        # Log initialization
        msg = f"[ParquetReader] __init__ called: thread_index={thread_index}, epoch={epoch}, dataset_type={dataset_type}"
        print(msg, file=sys.stderr, flush=True)
        self._logger.warning(msg)

    def _log_throughput(self, force=False):
        """Log samples per second and cache statistics on interval."""
        current_time = time.time()
        elapsed = current_time - self._last_log_time
        
        if force or elapsed >= self._log_interval_seconds:
            samples_per_second = self._sample_count / elapsed if elapsed > 0 else 0
            total_cache_accesses = self._cache_hits + self._cache_misses
            cache_hit_rate = (self._cache_hits / total_cache_accesses * 100) if total_cache_accesses > 0 else 0
            
            msg = (
                f"[ParquetReader thread={self.thread_index}] "
                f"samples/sec={samples_per_second:.1f}, "
                f"total_samples={self._total_samples_processed}, "
                f"interval_samples={self._sample_count}, "
                f"cache_hits={self._cache_hits}, "
                f"cache_misses={self._cache_misses}, "
                f"cache_hit_rate={cache_hit_rate:.1f}%"
            )
            
            # Use both print and logging to ensure visibility
            print(msg, file=sys.stderr, flush=True)
            self._logger.warning(msg)
            
            # Reset interval counters (but not total)
            self._sample_count = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._last_log_time = current_time

    @dlp.log
    def open(self, filename):
        """
        Open a Parquet file for lazy reading.
        
        Creates a memory-mapped ParquetFile handle for on-demand row group access.
        
        Args:
            filename: Path to the Parquet file
            
        Returns:
            ParquetFile handle for lazy access
        """
        msg = f"[ParquetReader thread={self.thread_index}] open() called: {filename}"
        print(msg, file=sys.stderr, flush=True)
        
        super().open(filename)
        self.parquet_file = pq.ParquetFile(filename)
        self.open_file_map[filename] = self.parquet_file
        
        # Log file metadata
        metadata = self.parquet_file.metadata
        msg = (f"[ParquetReader thread={self.thread_index}] opened: "
               f"num_rows={metadata.num_rows}, "
               f"num_row_groups={metadata.num_row_groups}, "
               f"num_columns={metadata.num_columns}")
        print(msg, file=sys.stderr, flush=True)
        
        return self.parquet_file

    @dlp.log
    def close(self, filename):
        """Close the Parquet file and clean up cached data."""
        if filename in self.open_file_map:
            # PyArrow ParquetFile doesn't need explicit close
            del self.open_file_map[filename]
        
        # Clear cache if it was for this file
        if self._cache_filename == filename:
            self._row_group_cache = None
            self._cache_row_group_idx = -1
            self._cache_filename = None

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Read a single sample from the Parquet file.
        
        Optimized for sequential access within files (file shuffle only).
        Uses row group caching for efficiency - consecutive samples in the
        same row group are served from cache.
        
        Args:
            filename: Path to the Parquet file
            sample_index: Index of the sample within the file
        """
        # Log first sample and every 10000th sample
        if self._total_samples_processed == 0 or self._total_samples_processed % 10000 == 0:
            msg = f"[ParquetReader thread={self.thread_index}] get_sample() sample_index={sample_index}, total_processed={self._total_samples_processed}"
            print(msg, file=sys.stderr, flush=True)
        
        super().get_sample(filename, sample_index)
        parquet_file = self.open_file_map[filename]
        
        # Determine which columns to read
        columns = [field for field in self.field_specs.keys()
                   if self.field_specs[field].get('read', True)]
        if not columns:
            columns = None  # Read all columns
        
        # Determine which row group contains this sample
        rows_per_group = parquet_file.metadata.row_group(0).num_rows
        row_group_idx = sample_index // rows_per_group
        row_idx = sample_index % rows_per_group
        
        # Check if we need to read a new row group
        if (self._cache_filename != filename or
            self._cache_row_group_idx != row_group_idx):
            # Log row group reads (actual I/O)
            if self._cache_misses < 10 or self._cache_misses % 100 == 0:
                msg = f"[ParquetReader thread={self.thread_index}] Reading row_group={row_group_idx} from {filename}"
                print(msg, file=sys.stderr, flush=True)
            
            # Read entire row group and cache it (actual I/O from disk)
            self._row_group_cache = parquet_file.read_row_group(
                row_group_idx,
                columns=columns
            )
            self._cache_row_group_idx = row_group_idx
            self._cache_filename = filename
            self._cache_misses += 1
        else:
            self._cache_hits += 1
        
        # Get the specific row from the cached row group
        row_data = self._row_group_cache.slice(row_idx, 1)
        
        # Calculate total size from the PyArrow table
        total_size = 0
        for column_name in row_data.column_names:
            column = row_data.column(column_name)
            total_size += column.nbytes
        
        dlp.update(image_size=total_size)
        
        # Increment sample count and check if we should log throughput
        self._sample_count += 1
        self._total_samples_processed += 1
        self._log_throughput()
        
        # Return the PyArrow table slice
        return row_data

    def next(self):
        """
        Iterator-based reading - delegates to parent class
        which calls get_sample for each sample.
        """
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, global_sample_idx, step):
        """
        Index-based reading - overrides parent class to keep files open.
        
        The base class FormatReader.read_index() opens and closes files for
        every sample when read_type is ON_DEMAND, which invalidates our
        row-group cache and causes 0% cache hit rate. This override keeps
        files open to enable proper caching.
        
        Args:
            global_sample_idx: Global index of the sample to read
            step: Current training step
            
        Returns:
            The resized image/sample data
        """
        self.step = step
        self.image_idx = global_sample_idx
        
        # Map global index to filename and sample index within file
        filename, sample_index = self.global_index_map[global_sample_idx]
        
        # Increment read counter
        FormatReader.read_images += 1
        
        # Open file only if not already open (keep files open for caching)
        if filename not in self.open_file_map or self.open_file_map[filename] is None:
            self.open_file_map[filename] = self.open(filename)
        
        # Read the sample (uses row-group cache)
        self.get_sample(filename, sample_index)
        
        # Apply preprocessing
        self.preprocess()
        
        # NOTE: We intentionally do NOT close the file here, unlike the base class
        # which closes on every sample when read_type is ON_DEMAND. Keeping files
        # open allows our row-group cache to work properly.
        
        return self._args.resized_image

    @dlp.log
    def finalize(self):
        """Clean up resources and log final statistics."""
        # Log final statistics before cleanup
        total_cache_accesses = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_cache_accesses * 100) if total_cache_accesses > 0 else 0
        self._logger.info(
            f"[ParquetReader thread={self.thread_index}] Finalizing - "
            f"total_samples_processed={self._total_samples_processed}, "
            f"final_cache_hits={self._cache_hits}, "
            f"final_cache_misses={self._cache_misses}, "
            f"final_cache_hit_rate={cache_hit_rate:.1f}%"
        )
        
        # Clear cache
        self._row_group_cache = None
        self._cache_row_group_idx = -1
        self._cache_filename = None
        
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
