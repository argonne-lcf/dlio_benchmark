"""
   Copyright (c) 2022, UChicago Argonne, LLC
   All Rights Reserved

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
        elif type == FormatType.JPEG:
            from src.data_generator.jpeg_generator import JPEGGenerator
            return JPEGGenerator()
        elif type == FormatType.PNG:
            from src.data_generator.png_generator import PNGGenerator
            return PNGGenerator()
        else:
            raise Exception(str(ErrorCodes.EC1001))