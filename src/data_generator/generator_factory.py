from src.common.enumerations import FormatType
from src.common.error_code import ErrorCodes



class GeneratorFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_generator(type):
        if type == FormatType.TFRECORD:
            from src.data_generator.tf_generator import TFRecordGenerator
            return TFRecordGenerator()
        elif type == FormatType.HDF5:
            from src.data_generator.hdf5_generator import HDF5Generator
            return HDF5Generator()
        elif type == FormatType.CSV:
            from src.data_generator.csv_generator import CSVGenerator
            return CSVGenerator()
        elif type == FormatType.NPZ:
            from src.data_generator.npz_generator import NPZGenerator
            return NPZGenerator()
        else:
            raise Exception(str(ErrorCodes.EC1001))