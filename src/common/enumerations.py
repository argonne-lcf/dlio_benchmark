from enum import Enum


class ComputationType(Enum):
    NONE = 0
    SYNC = 1
    ASYNC = 2

class FormatType(Enum):
    TFRECORD = 0
    HDF5 = 1
    CSV = 2
    NPZ = 3

class Profiler(Enum):
    NONE = 0
    DARSHAN = 1
    TENSORBOARD = 2