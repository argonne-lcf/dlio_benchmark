"""
Copyright (c) 2026, UChicago Argonne, LLC
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

import logging
import multiprocessing

import numpy as np

from dlio_benchmark.common.enumerations import DatasetType
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI


class _StubReader(FormatReader):
    """Reader with the file access stubbed out, so only the batching logic runs."""

    def open(self, filename):
        return filename

    def close(self, filename):
        pass

    def get_sample(self, filename, sample_index):
        return

    def read_index(self, image_idx, step):
        return

    def is_index_based(self):
        return False

    def is_iterator_based(self):
        return True

    def next(self):
        for batch in super().next():
            yield batch

    def finalize(self):
        pass


def _last_batch_length(batch_size, total_images, result):
    DLIOMPI.get_instance().initialize()
    args = ConfigArguments.get_instance()
    args.my_rank = 0
    args.comm_size = 1
    args.batch_size = batch_size
    args.preprocess_time = 0.0
    args.num_samples_per_file = total_images
    args.resized_image = np.zeros(1)
    args.file_list_train = ["file_0"]
    args.train_global_index_map = {}
    args.train_file_map = {
        0: [(index, "file_0", index) for index in range(total_images)]
    }

    reader = _StubReader(DatasetType.TRAIN, 0)
    batches = [len(batch) for batch in reader.next()]
    result.put(batches)


def _run_with_timeout(batch_size, total_images, timeout=30):
    result = multiprocessing.Queue()
    worker = multiprocessing.Process(
        target=_last_batch_length, args=(batch_size, total_images, result)
    )
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        worker.terminate()
        worker.join()
        return None
    return result.get(timeout=5)


def test_last_batch_is_padded_for_a_small_batch_size():
    batches = _run_with_timeout(4, 3)
    assert batches == [4], f"expected one padded batch of 4, got {batches}"


def test_last_batch_is_padded_for_a_batch_size_above_the_small_int_cache():
    """CPython interns small ints, so `len(batch) is not self.batch_size` only
    terminates while batch_size <= 256. Above that the padding loop never ends."""
    logging.info("padding a batch of 300 from 299 samples")
    batches = _run_with_timeout(300, 299)
    assert batches is not None, "the padding loop did not terminate"
    assert batches == [300], f"expected one padded batch of 300, got {batches}"
