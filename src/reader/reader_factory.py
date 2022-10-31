from src.common.enumerations import FormatType
from src.common.error_code import ErrorCodes


class ReaderFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_format(type):
        if type == FormatType.TFRECORD:
            from src.reader.tf_reader import TFReader
            return TFReader()
        elif type == FormatType.HDF5:
            from src.reader.hdf5_format import HDF5Reader
            return HDF5Reader()
        elif type == FormatType.HDF5_OPT:
            from src.reader.hdf5_stimulate import HDF5StimulateReader
            return HDF5StimulateReader()
        elif type == FormatType.CSV:
            from src.reader.csv_reader import CSVReader
            return CSVReader()
        elif type == FormatType.NPZ:
            from src.reader.npz_reader import NPZReader
            return NPZReader()
        elif type == FormatType.DATA_LOADER:
            from src.reader.data_loader_reader import DataLoaderReader
            return DataLoaderReader()
        else:
            raise Exception(str(ErrorCodes.EC1001))