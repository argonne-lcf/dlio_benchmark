"""
   Copyright (c) 2022, UChicago Argonne, LLC
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
import logging
import numpy as np
import tensorflow as tf
from PIL import Image

from numpy import random
from src.reader.reader_handler import FormatReader
from src.common.enumerations import Shuffle, FileAccess, DatasetType

from src.utils.utility import progress, utcnow

class PNGReader(FormatReader):
    """
    Reader for PNG files
    """
    def __init__(self, dataset_type):
        super().__init__(dataset_type)

    def read(self, epoch_number):
        """
        for each epoch it opens the npz files and reads the data into memory
        :param epoch_number:
        """
        super().read(epoch_number)
        self._dataset = self._local_file_list
        self.after_read()

    def next(self):
        """
        The iterator of the dataset just performs memory sub-setting for each portion of the data.
        :return: piece of data for training.
        """
        super().next()
        total = int(math.ceil(self.get_sample_len() / self.batch_size))
        count = 0
        batches_images = [self._dataset[n:n + self.batch_size] for n in range(0, len(self._dataset), self.batch_size)]
        total = len(batches_images)
        count = 0
        for batch in batches_images:
            count += 1
            images = []
            for filename in batch:
                images.append(np.asarray(Image.open(filename).resize((self.max_dimension, self.max_dimension))))
            images = np.array(images)
            is_last = 0 if count < total else 1
            logging.info(f"{utcnow()} completed {count} of {total} is_last {is_last} {len(self._dataset)}")
            yield is_last, images

    def read_index(self, index):
        return np.asarray(Image.open(self._dataset[index]).resize((self.max_dimension, self.max_dimension)))

    def get_sample_len(self):
        return self.num_samples * len(self._local_file_list)