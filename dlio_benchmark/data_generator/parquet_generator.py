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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress

# Map DLIO Compression enum values to PyArrow compression strings
COMPRESSION_MAP = {
    Compression.NONE: None,
    Compression.SNAPPY: 'snappy',
    Compression.GZIP: 'gzip',
    Compression.LZ4: 'lz4',
    Compression.ZSTD: 'zstd',
}


class ParquetGenerator(DataGenerator):
    """
    Schema-driven Parquet data generator with full compression and partitioning support.

    When parquet_columns is configured, generates multi-column files with specified
    dtypes (float32, float64, string, binary, bool). When empty, falls back to
    Phase 9 single 'data' column behavior for backward compatibility.

    Supports configurable row_group_size, batched writing for memory efficiency,
    and optional Hive-style partitioning.
    
    Memory Optimization Features:
    - Batched writing: Data is generated and written in batches to reduce peak memory usage
    - Vectorized Numpy-to-Arrow conversion: Uses FixedSizeListArray.from_arrays for zero-copy
      or near zero-copy conversion instead of inefficient list comprehensions
    - Configurable batch size via parquet_generation_batch_size parameter
    """

    def __init__(self):
        super().__init__()
        self.parquet_columns = self._args.parquet_columns
        self.row_group_size = self._args.parquet_row_group_size
        self.partition_by = self._args.parquet_partition_by
        # Use generation_batch_size if set, otherwise default to row_group_size
        self.generation_batch_size = self._args.parquet_generation_batch_size
        if self.generation_batch_size <= 0:
            self.generation_batch_size = self.row_group_size

    def _build_schema(self):
        """Build PyArrow schema from column specifications for use with ParquetWriter."""
        if not self.parquet_columns:
            # Backward compatible: single 'data' column with list of uint8
            return pa.schema([('data', pa.list_(pa.uint8()))])
        
        fields = []
        for col_spec in self.parquet_columns:
            if hasattr(col_spec, 'get'):
                name = str(col_spec.get('name', 'data'))
                dtype = str(col_spec.get('dtype', 'float32'))
                size = int(col_spec.get('size', 1024))
            else:
                name = str(col_spec)
                dtype = 'float32'
                size = 1024
            
            if dtype in ('float32', 'float64'):
                pa_inner = pa.float32() if dtype == 'float32' else pa.float64()
                # Use fixed size list for better memory efficiency
                fields.append(pa.field(name, pa.list_(pa_inner, size)))
            elif dtype == 'list':
                fields.append(pa.field(name, pa.list_(pa.float32(), size)))
            elif dtype == 'string':
                fields.append(pa.field(name, pa.string()))
            elif dtype == 'binary':
                fields.append(pa.field(name, pa.binary()))
            elif dtype == 'bool':
                fields.append(pa.field(name, pa.bool_()))
            else:
                # Fallback: treat unknown dtype as float32 list
                fields.append(pa.field(name, pa.list_(pa.float32(), size)))
        
        return pa.schema(fields)

    def _generate_column_data_batch(self, col_spec, batch_size):
        """
        Generate data for a single column based on its dtype specification.
        
        Uses optimized vectorized conversion for Numpy-to-Arrow to minimize
        memory overhead and avoid intermediate Python objects.
        """
        # Handle both dict and Hydra DictConfig by accessing values and casting to native types
        if hasattr(col_spec, 'get'):  # dict-like (dict or DictConfig)
            name = str(col_spec.get('name', 'data'))
            dtype = str(col_spec.get('dtype', 'float32'))
            size = int(col_spec.get('size', 1024))
        else:
            name = str(col_spec)
            dtype = 'float32'
            size = 1024

        if dtype in ('float32', 'float64'):
            np_dtype = np.float32 if dtype == 'float32' else np.float64
            # Generate data as contiguous array
            data = np.random.rand(batch_size, size).astype(np_dtype)
            # Optimized conversion: use FixedSizeListArray.from_arrays for zero-copy
            flat_data = data.ravel()
            arrow_flat = pa.array(flat_data)
            arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, size)
            return name, arrow_data

        if dtype == 'list':
            # Treat like float32 with configurable size
            data = np.random.rand(batch_size, size).astype(np.float32)
            # Optimized conversion
            flat_data = data.ravel()
            arrow_flat = pa.array(flat_data)
            arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, size)
            return name, arrow_data

        if dtype == 'string':
            data = [f"text_{j}" for j in range(batch_size)]
            return name, pa.array(data, type=pa.string())

        if dtype == 'binary':
            data = [np.random.bytes(size) for _ in range(batch_size)]
            return name, pa.array(data, type=pa.binary())

        if dtype == 'bool':
            data = np.random.choice([True, False], batch_size)
            return name, pa.array(data, type=pa.bool_())

        # Fallback: treat unknown dtype as float32
        data = np.random.rand(batch_size, size).astype(np.float32)
        flat_data = data.ravel()
        arrow_flat = pa.array(flat_data)
        arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, size)
        return name, arrow_data

    def _generate_batch_columns(self, batch_size):
        """Generate all columns for a batch of samples."""
        columns = {}
        for col_spec in self.parquet_columns:
            name, arrow_data = self._generate_column_data_batch(col_spec, batch_size)
            columns[name] = arrow_data
        return columns

    def _generate_legacy_batch(self, dim1, dim2, batch_size):
        """
        Generate backward-compatible single 'data' column batch.
        
        Uses optimized conversion for the legacy format.
        """
        record = np.random.randint(255, size=dim1 * dim2, dtype=np.uint8)
        # Create batch_size copies of the record using numpy broadcasting
        records = np.tile(record, (batch_size, 1))
        # Optimized conversion using FixedSizeListArray
        flat_data = records.ravel()
        arrow_flat = pa.array(flat_data)
        arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, dim1 * dim2)
        return {'data': arrow_data}

    def _generate_column_data(self, col_spec, num_samples):
        """
        Generate data for a single column based on its dtype specification.
        
        This method is kept for backward compatibility but uses the optimized
        batch generation internally.
        """
        return self._generate_column_data_batch(col_spec, num_samples)

    def generate(self):
        """
        Generate parquet data files with config-driven schema or backward-compatible single column.
        
        Uses batched writing strategy to minimize memory usage:
        - Opens ParquetWriter with pre-defined schema
        - Generates data in batches of size `generation_batch_size`
        - Writes each batch immediately to disk
        - Closes writer when complete
        
        This approach significantly reduces peak memory usage for large files.
        """
        super().generate()
        np.random.seed(10)
        record_label = 0
        dim = self.get_dimension(self.total_files_to_generate)

        # Resolve compression from enum
        compression = COMPRESSION_MAP.get(self.compression, None)

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            progress(i + 1, self.total_files_to_generate, "Generating Parquet Data")

            out_path_spec = self.storage.get_uri(self._file_list[i])

            if self.partition_by:
                # Partitioned writes don't support streaming, use table-based approach
                # but still use optimized column generation
                if self.parquet_columns:
                    columns = self._generate_batch_columns(self.num_samples)
                    table = pa.table(columns)
                else:
                    dim1 = dim[2 * i]
                    dim2 = dim[2 * i + 1]
                    columns = self._generate_legacy_batch(dim1, dim2, self.num_samples)
                    table = pa.table(columns)
                
                pq.write_to_dataset(
                    table,
                    root_path=os.path.dirname(out_path_spec),
                    partition_cols=[self.partition_by],
                    compression=compression,
                    row_group_size=self.row_group_size,
                )
            else:
                # Use batched writing for memory efficiency
                schema = self._build_schema()
                
                # Ensure parent directory exists
                parent_dir = os.path.dirname(out_path_spec)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                
                with pq.ParquetWriter(out_path_spec, schema, compression=compression) as writer:
                    num_batches = (self.num_samples + self.generation_batch_size - 1) // self.generation_batch_size
                    
                    for batch_idx in range(num_batches):
                        batch_start = batch_idx * self.generation_batch_size
                        batch_end = min(batch_start + self.generation_batch_size, self.num_samples)
                        current_batch_size = batch_end - batch_start
                        
                        if self.parquet_columns:
                            columns = self._generate_batch_columns(current_batch_size)
                        else:
                            dim1 = dim[2 * i]
                            dim2 = dim[2 * i + 1]
                            columns = self._generate_legacy_batch(dim1, dim2, current_batch_size)
                        
                        batch_table = pa.table(columns)
                        writer.write_table(batch_table, row_group_size=self.row_group_size)
                        
                        # Log batch progress for large files
                        if num_batches > 1 and self.my_rank == 0:
                            self.logger.debug(
                                f"File {i+1}/{self.total_files_to_generate}: "
                                f"Wrote batch {batch_idx+1}/{num_batches} "
                                f"({current_batch_size} samples)"
                            )

        np.random.seed()
