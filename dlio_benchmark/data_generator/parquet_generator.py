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

    Supports configurable row_group_size and optional Hive-style partitioning.
    """

    def __init__(self):
        super().__init__()
        self.parquet_columns = self._args.parquet_columns
        self.row_group_size = self._args.parquet_row_group_size
        self.partition_by = self._args.parquet_partition_by

    def _generate_column_data(self, col_spec, num_samples):
        """Generate data for a single column based on its dtype specification."""
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
            pa_inner = pa.float32() if dtype == 'float32' else pa.float64()
            data = np.random.rand(num_samples, size).astype(np_dtype)
            arrow_data = pa.array([row.tolist() for row in data], type=pa.list_(pa_inner))
            return name, arrow_data

        if dtype == 'list':
            # Treat like float32 with configurable size
            data = np.random.rand(num_samples, size).astype(np.float32)
            arrow_data = pa.array([row.tolist() for row in data], type=pa.list_(pa.float32()))
            return name, arrow_data

        if dtype == 'string':
            data = [f"text_{j}" for j in range(num_samples)]
            return name, pa.array(data, type=pa.string())

        if dtype == 'binary':
            data = [np.random.bytes(size) for _ in range(num_samples)]
            return name, pa.array(data, type=pa.binary())

        if dtype == 'bool':
            data = np.random.choice([True, False], num_samples).tolist()
            return name, pa.array(data, type=pa.bool_())

        # Fallback: treat unknown dtype as float32
        data = np.random.rand(num_samples, size).astype(np.float32)
        arrow_data = pa.array([row.tolist() for row in data], type=pa.list_(pa.float32()))
        return name, arrow_data

    def generate(self):
        """
        Generate parquet data files with config-driven schema or backward-compatible single column.
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

            if self.parquet_columns:
                # Config-driven multi-column schema
                columns = {}
                for col_spec in self.parquet_columns:
                    name, arrow_data = self._generate_column_data(col_spec, self.num_samples)
                    columns[name] = arrow_data

                table = pa.table(columns)
            else:
                # Backward compatible: single 'data' column with random uint8
                dim1 = dim[2 * i]
                dim2 = dim[2 * i + 1]
                record = np.random.randint(255, size=dim1 * dim2, dtype=np.uint8)
                records = [record] * self.num_samples
                table = pa.table({'data': [rec.tolist() for rec in records]})

            if self.partition_by:
                pq.write_to_dataset(
                    table,
                    root_path=os.path.dirname(out_path_spec),
                    partition_cols=[self.partition_by],
                    compression=compression,
                    row_group_size=self.row_group_size,
                )
            else:
                pq.write_table(
                    table,
                    out_path_spec,
                    compression=compression,
                    row_group_size=self.row_group_size,
                )

        np.random.seed()
