from src.common.enumerations import FormatType
from src.common.error_code import ErrorCodes
from src.format.csv_format import CSVFormat
from src.format.hdf5_format import HDF5Format
from src.format.npz_format import NPZFormat
from src.format.tf_format import TFFormat


class FormatFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_format(type):
        if type == FormatType.TFRECORD:
            return TFFormat()
        elif type == FormatType.HDF5:
            return HDF5Format()
        elif type == FormatType.CSV:
            return CSVFormat()
        elif type == FormatType.NPZ:
            return NPZFormat()
        else:
            raise Exception(str(ErrorCodes.EC1001))