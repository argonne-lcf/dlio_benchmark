"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of DLProfile
 DLIO is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
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

class ReadType(Enum):
    """
    Type of read to be performed in the benchmark. 

    Q: Need to discuss this; I think when the dataset fits in memory, we are loading it. If it does not fit, it is more similar to on_demand.
    Huihuo: Yes, on_demand is loading data in a batch-by-batch fashion; where IN_MEMORY is loading data all at once in the beginning. 
    In most of the cases, data loader is used, so it is ON_DEMAND even when the dataset fits in memory. 
    → ON_DEMAND for most workloads; BERT might load everything; we need to have a way to scale the workloads up to obtain the dataset size:memory size ratio that we want.
    """
    IN_MEMORY = 'memory'
    ON_DEMAND = 'on_demand'

    def __str__(self):
        return self.value

class FileAccess(Enum):
    """
    File access mode.
    From Huihuo's comments:
    - "Multi = save dataset into multiple files
    - Shared = save everything in a single file
    - Collective = specific for the shared case, when we want to do collective I/O. 
    “Collective” is MPI specific; typically used for a huge file with small objects; 
    one thread T reads from disk and the other threads read from T's memory, which is used as a cache.
    "
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
