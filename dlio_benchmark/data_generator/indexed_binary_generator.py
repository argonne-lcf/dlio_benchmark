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

import struct

from mpi4py import MPI
import numpy as np

from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR
from dlio_benchmark.utils.utility import Profile, progress, utcnow, DLIOMPI, gen_random_tensor

dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in Indexed Binary format.
"""
class IndexedBinaryGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def index_file_path_off(self, prefix_path):
        return prefix_path + '.off.idx'

    def index_file_path_size(self, prefix_path):
        return prefix_path + '.sz.idx'

    @dlp.log
    def generate(self):
        """
        Generator for creating data in Indexed Binary format.

        Improvements over the original:
        - Each MPI rank uses a unique seed (``BASE_SEED + my_rank``) for
          collective I/O so different ranks produce different random data.
        - The individual I/O path uses per-file seeds (``BASE_SEED + i``) via
          ``gen_random_tensor`` so file content is reproducible across runs.
        """
        super().generate()

        # Rank-unique seed for the global numpy state (used by get_dimension
        # and collective I/O paths that use np.random.* directly).
        np.random.seed(self.BASE_SEED + self.my_rank)

        GiB = 1024 * 1024 * 1024
        samples_processed = 0
        total_samples = self.total_files_to_generate * self.num_samples
        dim = self.get_dimension(self.total_files_to_generate)

        if self.total_files_to_generate <= self.comm_size:
            # ── Collective I/O path ──────────────────────────────────────────
            samples_per_rank = (
                self.num_samples + (self.num_samples % self.comm_size)
            ) // self.comm_size

            for file_index in dlp.iter(range(int(self.total_files_to_generate))):
                amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
                comm = MPI.COMM_WORLD
                dim_ = dim[2 * file_index]
                shape_size = 0
                if isinstance(dim_, list):
                    shape_size = sum(dim_)
                else:
                    dim1 = dim_
                    dim2 = dim[2 * file_index + 1]
                    shape_size = dim1 * dim2
                sample_size = shape_size * self._args.record_element_bytes
                out_path_spec = self.storage.get_uri(self._file_list[file_index])
                out_path_spec_off_idx = self.index_file_path_off(out_path_spec)
                out_path_spec_sz_idx = self.index_file_path_size(out_path_spec)

                if self.my_rank == 0:
                    self.logger.info(f"{utcnow()} Starting metadata generation.")
                fh_off = MPI.File.Open(comm, out_path_spec_off_idx, amode)
                fh_sz = MPI.File.Open(comm, out_path_spec_sz_idx, amode)
                off_type = np.uint64
                elements_per_loop = min(
                    int(GiB / np.dtype(off_type).itemsize), samples_per_rank
                )
                offsets_processed = 0
                for element_index in range(
                    self.my_rank * samples_per_rank,
                    samples_per_rank * (self.my_rank + 1),
                    elements_per_loop,
                ):
                    offsets = np.array(
                        range(
                            self.my_rank * elements_per_loop * sample_size,
                            (self.my_rank + 1) * elements_per_loop * sample_size,
                            sample_size,
                        ),
                        dtype=off_type,
                    )
                    sizes = np.array([sample_size] * elements_per_loop, dtype=off_type)
                    offset = element_index * np.dtype(off_type).itemsize
                    fh_off.Write_at_all(offset, offsets)
                    fh_sz.Write_at_all(offset, sizes)
                    offsets_processed += elements_per_loop
                    progress(
                        offsets_processed * self.comm_size,
                        total_samples,
                        "Generating Indexed Binary Data Index for Samples",
                    )
                fh_off.Close()
                fh_sz.Close()

                if self.my_rank == 0:
                    self.logger.info(f"{utcnow()} Starting Sample generation.")

                fh = MPI.File.Open(comm, out_path_spec, amode)
                samples_per_loop = int(GiB / sample_size)

                # Rank-unique deterministic data for collective writes.
                records = gen_random_tensor(
                    shape=(sample_size * samples_per_loop,),
                    dtype=np.uint8,
                    seed=self.BASE_SEED + self.my_rank,
                )

                for sample_index in range(
                    self.my_rank * samples_per_rank,
                    samples_per_rank * (self.my_rank + 1),
                    samples_per_loop,
                ):
                    offset = sample_index * sample_size
                    fh.Write_at_all(offset, records)
                    samples_processed += samples_per_loop
                    progress(
                        samples_processed * self.comm_size,
                        total_samples,
                        "Generating Indexed Binary Data Samples",
                    )
                fh.Close()
        else:
            # ── Individual I/O path ──────────────────────────────────────────
            for i in dlp.iter(
                range(self.my_rank, int(self.total_files_to_generate), self.comm_size)
            ):
                dim_ = dim[2 * i]
                shape_size = 0
                if isinstance(dim_, list):
                    shape_size = int(np.prod(dim_))
                else:
                    dim1 = dim_
                    dim2 = dim[2 * i + 1]
                    shape_size = dim1 * dim2
                sample_size = shape_size * self._args.record_element_bytes
                total_size = sample_size * self.num_samples
                write_size = total_size
                memory_size = self._args.generation_buffer_size
                if total_size > memory_size:
                    write_size = memory_size - (memory_size % sample_size)

                out_path_spec = self.storage.get_uri(self._file_list[i])
                out_path_spec_off_idx = self.index_file_path_off(out_path_spec)
                out_path_spec_sz_idx = self.index_file_path_size(out_path_spec)
                progress(i + 1, self.total_files_to_generate, "Generating Indexed Binary Data")

                data_file = open(out_path_spec, "wb")
                off_file = open(out_path_spec_off_idx, "wb")
                sz_file = open(out_path_spec_sz_idx, "wb")

                # Per-file deterministic data via gen_random_tensor.
                records = gen_random_tensor(
                    shape=(write_size,),
                    dtype=np.uint8,
                    seed=self._file_seed(i),
                )

                written_bytes = 0
                while written_bytes < total_size:
                    data_to_write = (
                        write_size
                        if written_bytes + write_size <= total_size
                        else total_size - written_bytes
                    )
                    samples_to_write = data_to_write // sample_size

                    # Write data
                    myfmt = 'B' * data_to_write
                    binary_data = struct.pack(myfmt, *records[:data_to_write])
                    data_file.write(binary_data)
                    struct._clearcache()

                    # Write offsets
                    myfmt = 'Q' * samples_to_write
                    offsets = range(0, data_to_write, sample_size)
                    offsets = offsets[:samples_to_write]
                    binary_offsets = struct.pack(myfmt, *offsets)
                    off_file.write(binary_offsets)

                    # Write sizes
                    myfmt = 'Q' * samples_to_write
                    sample_sizes = [sample_size] * samples_to_write
                    binary_sizes = struct.pack(myfmt, *sample_sizes)
                    sz_file.write(binary_sizes)

                    written_bytes = written_bytes + data_to_write

                data_file.close()
                off_file.close()
                sz_file.close()
            np.random.seed()

        DLIOMPI.get_instance().comm().Barrier()
