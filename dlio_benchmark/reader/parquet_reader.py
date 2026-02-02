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
import pyarrow as pa
import pyarrow.parquet as pq

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.reader.reader_handler import FormatReader

dlp = Profile(MODULE_DATA_READER)


class ParquetReader(FormatReader):
    """
    Memory-efficient Parquet reader with column filtering and schema validation.

    Supports two read modes:
    - "default": Reads entire file (with column filtering) into numpy array via memory-mapped I/O.
    - "row_group": Reads file row-group by row-group for memory-efficient decompression,
      then concatenates into numpy array for DLIO's index-based access pattern.

    When parquet_columns is configured, only those columns are read from disk.
    Schema validation ensures requested columns exist in the file before reading.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        # Extract column names from config; empty list means read all columns
        self.column_names = []
        if self._args.parquet_columns:
            self.column_names = [
                col['name'] if isinstance(col, dict) else col
                for col in self._args.parquet_columns
            ]
        self.read_mode = self._args.parquet_read_mode

    def _validate_schema(self, filename):
        """Validate that all requested columns exist in the file schema."""
        if not self.column_names:
            return
        file_schema = pq.read_schema(filename)
        file_column_names = set(file_schema.names)
        missing = [c for c in self.column_names if c not in file_column_names]
        if missing:
            raise ValueError(
                f"Schema mismatch in '{filename}': requested columns {missing} "
                f"not found in file schema. Available columns: {sorted(file_column_names)}"
            )

    @dlp.log
    def open(self, filename):
        super().open(filename)
        self._validate_schema(filename)

        columns = self.column_names if self.column_names else None

        if self.read_mode == 'row_group':
            # Read row-group by row-group for memory-efficient decompression
            pf = pq.ParquetFile(filename, memory_map=True)
            batches = []
            for batch in pf.iter_batches(columns=columns):
                batches.append(batch)
            table = pa.Table.from_batches(batches)
        else:
            # Default mode: read entire table with column filtering
            table = pq.read_table(filename, columns=columns, memory_map=True)

        return table.to_pandas().to_numpy()

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        image = self.open_file_map[filename][sample_index]
        dlp.update(image_size=image.nbytes)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
