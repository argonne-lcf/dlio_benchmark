from src.common.enumerations import FormatType
from src.common.error_code import ErrorCodes
from src.data_generator.csv_generator import CSVGenerator
from src.data_generator.tf_generator import TFRecordGenerator
from src.data_generator.hdf5_generator import HDF5Generator
from src.data_generator.npz_generator import NPZGenerator
from src.data_generator.jpeg_generator import JPEGGenerator
from src.data_generator.png_generator import PNGGenerator



class GeneratorFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_generator(type):
        if type == FormatType.TFRECORD:
            return TFRecordGenerator()
        elif type == FormatType.HDF5:
            return HDF5Generator()
        elif type == FormatType.CSV:
            return CSVGenerator()
        elif type == FormatType.NPZ:
            return NPZGenerator()
        elif type == FormatType.JPEG:
            return JPEGGenerator()
        elif type == FormatType.PNG:
            return PNGGenerator()
        else:
            raise Exception(str(ErrorCodes.EC1001))