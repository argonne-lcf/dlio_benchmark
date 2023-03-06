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

from time import time
import math
import logging
import numpy as np
from PIL import Image
from src.reader.reader_handler import FormatReader
from src.utils.utility import progress, utcnow, timeit, perftrace

class JPEGReader(FormatReader):
    """
    Reader for JPEG files
    """
    def __init__(self, dataset_type):
        super().__init__(dataset_type)


    @perftrace.event_logging
    def read(self, epoch_number):
        """
        for each epoch it opens the npz files and reads the data into memory
        :param epoch_number:
        """
        super().read(epoch_number)
        self._dataset = self._local_file_list
        self.after_read()

    @perftrace.event_logging
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
        image_index = 0
        for batch in batches_images:
            images = []
            for filename in batch:
                t0 = time()
                images.append(np.asarray(Image.open(filename).resize((self.max_dimension, self.max_dimension))))
                t1 = time()
                perftrace.event_complete(f"JPEG_{self.dataset_type}_image_{image_index}_step_{count}",
                                         "csv_reader..next", t0, t1 - t0)
                t0 = time()
                image_index += 1
            images = np.array(images)
            is_last = 0 if count < total else 1
            count += 1
            logging.info(f"{utcnow()} completed {count} of {total} is_last {is_last} {len(self._dataset)}")
            yield is_last, images

    @perftrace.event_logging
    def read_index(self, index):
        return np.asarray(Image.open(self._dataset[index]).resize((self.max_dimension, self.max_dimension)))

    @perftrace.event_logging
    def get_sample_len(self):
        return self.num_samples * len(self._local_file_list)