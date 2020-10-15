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
