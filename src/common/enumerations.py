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

from enum import Enum

class FrameworkType(Enum):
    """
    Different Computation Type for training loop.
    """
    TENSORFLOW = 'tensorflow'
    PYTORCH = 'pytorch'

    def __str__(self):
        return self.value

class ComputationType(Enum):
    """
    Different Computation Type for training loop.
    """
    NONE = 'none'
    SYNC = 'sync'
    ASYNC = 'async'


class FormatType(Enum):
    """
    Format Type supported by the benchmark.
    """
    TFRECORD = 'tfrecord'
    HDF5 = 'hdf5'
    CSV = 'csv'
    NPZ = 'npz'
    HDF5_OPT = 'hdf5_opt'
    JPEG = 'jpeg'
    PNG = 'png'
    WAV = 'wav'

    def __str__(self):
        return self.value

class DataLoaderType(Enum):
    """
    Framework DataLoader Type
    """
    TENSORFLOW='tensorflow'
    PYTORCH='pytorch'

    def __str__(self):
        return self.value

class Profiler(Enum):
    """
    Profiler types supported by the benchmark.
    """
    NONE = 'none'
    IOSTAT = 'iostat'
    DARSHAN = 'darshan'
    TENSORBOARD = 'tensorboard'

    def __str__(self):
        return self.value

class Shuffle(Enum):
    """
    Shuffle mode for files and memory.
    """
    OFF = 'off'
    SEED = 'seed'
    RANDOM = 'random'

    def __str__(self):
        return self.value

class ReadType(Enum):
    """
    Type of read to be performed in the benchmark. 
    - On Demand: loading data in a batch-by-batch fashion
    - In Memory: loading data all at once in the beginning. 
    """
    IN_MEMORY = 'memory'
    ON_DEMAND = 'on_demand'

    def __str__(self):
        return self.value

class FileAccess(Enum):
    """
    File access mode.
    - Multi = save dataset into multiple files
    - Shared = save everything in a single file
    - Collective = specific for the shared case, when we want to do collective I/O. Typically used for a huge file with small objects. 
      One thread T reads from disk and the other threads read from T's memory, which is used as a cache.
    """
    MULTI = 'multi'
    SHARED = 'shared'
    COLLECTIVE = 'collective'

    def __str__(self):
        return self.value


class Compression(Enum):
    """
    Different Compression Libraries.
    """
    NONE = 'none'
    GZIP = 'gzip'
    LZF = 'lzf'
    BZIP2 = 'bz2'
    ZIP = 'zip'
    XZ = 'xz'

    def __str__(self):
        return self.value
