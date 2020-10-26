from src.common.enumerations import FormatType
from src.common.error_code import ErrorCodes
from src.reader.csv_reader import CSVReader
from src.reader.hdf5_format import HDF5Reader
from src.reader.hdf5_stimulate import HDF5StimulateReader
from src.reader.npz_reader import NPZReader
from src.reader.tf_reader import TFReader


class ReaderFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_format(type):
        if type == FormatType.TFRECORD:
            return TFReader()
        elif type == FormatType.HDF5:
            return HDF5Reader()
        elif type == FormatType.HDF5_OPT:
            return HDF5StimulateReader()
        elif type == FormatType.CSV:
            return CSVReader()
        elif type == FormatType.NPZ:
            return NPZReader()
        else:
            raise Exception(str(ErrorCodes.EC1001))