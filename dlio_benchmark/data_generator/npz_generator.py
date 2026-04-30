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

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile, progress, gen_random_tensor
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

# Fast path: use s3dlio Rust NPZ generator when available.
# Eliminates Python's software CRC32 bottleneck (~178 ms for 140 MiB)
# by using crc32fast (hardware-accelerated, ~20 ms) and Rayon in-place fill.
try:
    import s3dlio as _s3dlio
    _HAS_S3DLIO_NPZ = hasattr(_s3dlio, "generate_npz_bytes")
except ImportError:
    _HAS_S3DLIO_NPZ = False

"""
Generator for creating data in NPZ format.
"""
class NPZGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    @dlp.log
    def generate(self):
        """
        Generator for creating data in NPZ format of 3d dataset.
        Uses the base-class template for seeding, BytesIO, and put_data.
        Bug fix: pass output.getvalue() (bytes) to put_data, not the BytesIO object.
        """
        super().generate()
        dtype = self._args.record_element_dtype
        num_samples = self.num_samples
        record_labels = [0] * num_samples
        compression = self.compression

        def _write(i, dim_, dim1, dim2, file_seed, rng,
                   out_path_spec, is_local, output):
            # ── Fast path: Rust NPZ builder ──────────────────────────────
            # Only for STORED (no compression) — same condition as np.savez.
            # Saves ~5x over np.savez by eliminating software CRC32 and
            # the intermediate DataBuffer allocation.
            if _HAS_S3DLIO_NPZ and compression != Compression.ZIP:
                shape = (list(dim_) + [num_samples]
                         if isinstance(dim_, list)
                         else [dim1, dim2, num_samples])
                dtype_str = np.dtype(dtype).str  # e.g. '<f4' for float32
                npz_view = _s3dlio.generate_npz_bytes(
                    shape=shape, dtype=dtype_str, num_samples=num_samples)
                if is_local:
                    with open(output, "wb") as f:
                        f.write(npz_view)
                else:
                    output.write(npz_view)
                return
            # ── Slow path: numpy fallback ─────────────────────────────────
            if isinstance(dim_, list):
                records = gen_random_tensor(
                    shape=(*dim_, num_samples), dtype=dtype,
                    rng=rng, writeable=False)
            else:
                records = gen_random_tensor(
                    shape=(dim1, dim2, num_samples), dtype=dtype,
                    rng=rng, writeable=False)
            if compression != Compression.ZIP:
                np.savez(output, x=records, y=record_labels)
            else:
                np.savez_compressed(output, x=records, y=record_labels)
            # Note: template calls output.getvalue() for object storage — bug fixed.

        self._generate_files(_write, "NPZ Data")
