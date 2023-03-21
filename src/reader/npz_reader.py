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
import numpy as np
from src.common.constants import MODULE_DATA_READER
from src.reader.reader_handler import FormatReader
from src.utils.utility import Profile

class NPZReader(FormatReader):
    """
    Reader for NPZ files
    """

    def __init__(self, dataset_type, thread_index, epoch_number):
        with Profile(name=f"{self.__init__.__qualname__}", cat=MODULE_DATA_READER, epoch=epoch_number):
            super().__init__(dataset_type, thread_index, epoch_number)

    def open(self, filename):
        with Profile(name=f"{self.open.__qualname__}", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                     image_idx=self.image_idx):
            super().open(filename)
            return np.load(filename, allow_pickle=True)["x"]

    def close(self, filename):
        with Profile(name=f"{self.close.__qualname__}", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                     image_idx=self.image_idx):
            super().close(filename)

    def get_sample(self, filename, sample_index):
        with Profile(name=f"{self.get_sample.__qualname__}", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                     image_idx=self.image_idx):
            super().get_sample(filename, sample_index)
            with Profile(name=f"{self.get_sample.__qualname__}.read", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                         image_idx=self.image_idx) as p:
                my_image = self.open_file_map[filename][..., sample_index]
                p.update(image_size=my_image.nbytes)
            with Profile(name=f"{self.get_sample.__qualname__}.resize", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                         image_idx=self.image_idx) as p:
                self.preprocess()
                #resized_image = np.resize(my_image, (self._args.max_dimension, self._args.max_dimension))
                resized_image = random.random((self._args.max_dimension, self._args.max_dimension))
                p.update(image_size=resized_image.nbytes)
            return resized_image

    def next(self):
        step = 1
        with Profile(name=f"{self.next.__qualname__}", cat=MODULE_DATA_READER, ) as lp:
            for is_last, batch in super().next():
                lp.update(epoch=self.epoch_number, step=step).flush()
                yield is_last, batch
                step += 1
                lp.reset()

    def read_index(self, index):
        with Profile(name=f"{self.read_index.__qualname__}", cat=MODULE_DATA_READER, epoch=self.epoch_number,
                     image_idx=index):
            return super().read_index(index)

    def finalize(self):
        with Profile(name=f"{self.finalize.__qualname__}", cat=MODULE_DATA_READER, epoch=self.epoch_number):
            return super().finalize()
