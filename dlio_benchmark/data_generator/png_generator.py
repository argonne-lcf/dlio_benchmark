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
import io
import numpy as np
import PIL.Image as im

from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.utils.utility import progress, utcnow, gen_random_tensor
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

try:
    import dgen_py as _dgen_py
    _HAS_DGEN = True
except ImportError:
    _dgen_py = None
    _HAS_DGEN = False

dlp = Profile(MODULE_DATA_GENERATOR)

class PNGGenerator(DataGenerator):
    @dlp.log
    def generate(self):
        """
        Generator for creating data in PNG format of 3d dataset.
        Uses the base-class template for seeding, BytesIO, and put_data.

        Fast path (non-DALI): streams raw random bytes via a single dgen_py
        Generator created once before the file loop — same producer pattern
        as StreamingCheckpointing.  The generator advances continuously from
        one image to the next; no reset, no re-seed, always fresh data.
        A single bytearray is pre-allocated and reused for every image;
        only one copy occurs (bytearray → BytesIO / file).

        DALI path: keeps the full PIL encode because fn.decoders.image()
        requires a valid PNG bitstream.
        """
        super().generate()
        my_rank = self.my_rank
        total = self.total_files_to_generate
        logger = self.logger
        use_fast_path = (self._args.data_loader != DataLoaderType.NATIVE_DALI)

        # --- One streaming generator per process, created ONCE before the loop ---
        # Sized at 256 GiB — far larger than any realistic dataset so it never
        # exhausts during a single datagen run.  fill_chunk() keeps streaming
        # continuously; we never call reset() or set_seed().
        _stream = _dgen_py.Generator(size=256 * 1024 ** 3) if _HAS_DGEN else None
        _buf = None      # bytearray reused each image (lazy-allocated / grown)
        _buf_size = 0    # current capacity of _buf

        def _write(i, dim_, dim1, dim2, file_seed, rng,
                   out_path_spec, is_local, output):
            nonlocal _buf, _buf_size
            nbytes = dim1 * dim2
            if my_rank == 0:
                logger.debug(f"{utcnow()} Dimension of images: {dim1} x {dim2}")
            if my_rank == 0 and i % 100 == 0:
                logger.info(f"Generated file {i}/{total}")
            if use_fast_path:
                if _stream is not None and not is_local:
                    # Object-store async pipeline: _write is called from the
                    # main thread only — generator access is single-threaded.
                    # Grow the reuse buffer only when a larger image appears.
                    if nbytes > _buf_size:
                        _buf = bytearray(nbytes)
                        _buf_size = nbytes
                    mv = memoryview(_buf)[:nbytes]
                    _stream.fill_chunk(mv)
                    output.write(mv)
                else:
                    # Local-FS thread-pool path: multiple threads call _write
                    # concurrently — use the thread-safe gen_random_tensor.
                    data = gen_random_tensor(shape=(dim1, dim2), dtype=np.uint8,
                                            rng=rng, writeable=False)
                    if is_local:
                        with open(out_path_spec, 'wb') as f:
                            f.write(data)
                    else:
                        output.write(data)
            else:
                # DALI path: PIL encode required for fn.decoders.image().
                if _stream is not None:
                    if nbytes > _buf_size:
                        _buf = bytearray(nbytes)
                        _buf_size = nbytes
                    mv = memoryview(_buf)[:nbytes]
                    _stream.fill_chunk(mv)
                    records = np.frombuffer(mv, dtype=np.uint8).reshape(dim1, dim2)
                else:
                    records = gen_random_tensor(shape=(dim1, dim2), dtype=np.uint8, rng=rng)
                img = im.fromarray(records)
                img.save(output, format='PNG')

        self._generate_files(_write, "PNG Data")
