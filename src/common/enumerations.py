from enum import Enum


class ComputationType(Enum):
    NONE = 'none'
    SYNC = 'sync'
    ASYNC = 'async'


class FormatType(Enum):
    TFRECORD = 'tfrecord'
    HDF5 = 'hdf5'
    CSV = 'csv'
    NPZ = 'npz'

    def __str__(self):
        return self.value


class Profiler(Enum):
    NONE = 'none'
    DARSHAN = 'darshan'
    TENSORBOARD = 'tensorboard'

    def __str__(self):
        return self.value


class Shuffle(Enum):
    OFF = 'off'
    SEED = 'seed'
    RANDOM = 'random'

    def __str__(self):
        return self.value


class ReadType(Enum):
    IN_MEMORY = 'memory'
    ON_DEMAND = 'on_demand'

    def __str__(self):
        return self.value


class FileAccess(Enum):
    MULTI = 'multi'
    SHARED = 'shared'
    COLLECTIVE = 'collective'

    def __str__(self):
        return self.value

class Compression(Enum):
    NONE = 'none'
    GZIP = 'gzip'
    LZF = 'lzf'

    def __str__(self):
        return self.value
