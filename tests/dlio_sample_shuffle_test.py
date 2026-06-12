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

import logging

from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
from dlio_benchmark.common.enumerations import Shuffle

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_sample_shuffle_test.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    format="[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]",
)


def init():
    DLIOMPI.get_instance().initialize()


def test_sample_shuffle_bug():
    init()
    logging.info("=" * 80)
    logging.info(" DLIO test for sample_shuffle bug reproduction")
    logging.info("=" * 80)

    args = ConfigArguments.get_instance()
    args.num_files_train = 2
    args.num_samples_per_file = 4
    args.sample_shuffle = Shuffle.SEED
    args.seed = 42
    args.seed_change_epoch = True
    args.comm_size = 1
    args.my_rank = 0
    args.read_threads = 1

    result, _ = args.build_sample_map_iter(["file_0", "file_1"], 8, epoch_number=0)

    samples = {"file_0": [], "file_1": []}
    for _, entries in result.items():
        for _, path, idx in entries:
            samples[path.split("/")[-1]].append(idx)

    logging.info(f"Result:")
    logging.info(f"  file_0: {samples['file_0']}")
    logging.info(f"  file_1: {samples['file_1']}")

    expected = [0, 1, 2, 3]
    assert (
        sorted(samples["file_0"]) == expected
    ), f"file_0 should have {expected}, got {sorted(samples['file_0'])}"
    assert (
        sorted(samples["file_1"]) == expected
    ), f"file_1 should have {expected}, got {sorted(samples['file_1'])}"


if __name__ == "__main__":
    test_sample_shuffle_bug()
