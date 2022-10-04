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

"""
Different Computation Type for training loop.
"""
class ComputationType(Enum):
    NONE = 'none'
    SYNC = 'sync'
    ASYNC = 'async'

"""
Format Type supported by the benchmark.
"""
class FormatType(Enum):
    TFRECORD = 'tfrecord'
    HDF5 = 'hdf5'
    CSV = 'csv'
    NPZ = 'npz'
    HDF5_OPT = 'hdf5_opt'

    def __str__(self):
        return self.value

"""
Profiler types supported by the benchmark.
"""
class Profiler(Enum):
    NONE = 'none'
    DARSHAN = 'darshan'
    TENSORBOARD = 'tensorboard'

    def __str__(self):
        return self.value

"""
Shuffle mode for files and memory.
"""
class Shuffle(Enum):
    OFF = 'off'
    SEED = 'seed'
    RANDOM = 'random'

    def __str__(self):
        return self.value

"""
Type of read to be performed in the benchmark. 
"""
class ReadType(Enum):
    IN_MEMORY = 'memory'
    ON_DEMAND = 'on_demand'

    def __str__(self):
        return self.value

"""
File access mode.
"""
class FileAccess(Enum):
    MULTI = 'multi'
    SHARED = 'shared'
    COLLECTIVE = 'collective'

    def __str__(self):
        return self.value


"""
Different Compression Libraries.
"""
class Compression(Enum):
    NONE = 'none'
    GZIP = 'gzip'
    LZF = 'lzf'
    BZIP2 = 'bz2'
    ZIP = 'zip'
    XZ = 'xz'

    def __str__(self):
        return self.value
