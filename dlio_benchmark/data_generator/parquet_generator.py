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

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dlio_benchmark.utils.utility import progress


class ParquetGenerator(DataGenerator):
    """
    Generator for creating data in Parquet format.
    """

    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generate parquet data for training. It generates a 2d dataset and writes it to file.
        """
        super().generate()
        np.random.seed(10)
        record_label = 0
        dim = self.get_dimension(self.total_files_to_generate)

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            progress(i + 1, self.total_files_to_generate, "Generating Parquet Data")
            dim1 = dim[2 * i]
            dim2 = dim[2 * i + 1]

            # Generate random data as numpy array (like CSV generator)
            record = np.random.randint(255, size=dim1 * dim2, dtype=np.uint8)
            records = [record] * self.num_samples

            # Convert to Arrow table
            table = pa.table({'data': [rec.tolist() for rec in records]})

            out_path_spec = self.storage.get_uri(self._file_list[i])

            # Map DLIO compression to parquet compression
            compression = 'snappy'  # default for parquet
            if self.compression == Compression.GZIP:
                compression = 'gzip'
            elif self.compression == Compression.NONE:
                compression = None

            pq.write_table(table, out_path_spec, compression=compression)

        np.random.seed()
