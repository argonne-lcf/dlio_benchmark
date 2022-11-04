"""
   Copyright 2021 UChicago Argonne, LLC

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

from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator

import logging
import numpy as np
from numpy import random

from src.utils.utility import progress
from shutil import copyfile
import PIL.Image as im
"""
Generator for creating data in JPEG format.
"""
class JPEGGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generator for creating data in NPZ format of 3d dataset.
        """
        super().generate()
        dim = int(np.sqrt(self.record_size/3.0))
        record_labels = [0] 
        prev_out_spec =""
        count = 0
        logging.info(f"dimension of images: {dim} x {dim} x 3")

        for i in range(0, int(self.total_files_to_generate)):
            records = random.randint(255, size=(dim, dim, 3), dtype=np.uint8)
            img = im.fromarray(records)
            if self.my_rank == 0 and i % 100 == 0:
                logging.info(f"Generated file {i}/{self.total_files_to_generate}")
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.total_files_to_generate, "Generating JPEG Data")
                out_path_spec = "{}_{}_of_{}.jpeg".format(self._file_prefix, i, self.total_files_to_generate)
                if count == 0:
                    prev_out_spec = out_path_spec
                    img.save(out_path_spec, format='JPEG', bits=8)
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)