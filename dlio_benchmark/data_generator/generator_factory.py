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

from dlio_benchmark.common.enumerations import FormatType
from dlio_benchmark.common.error_code import ErrorCodes



class GeneratorFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_generator(type):
        if type in [FormatType.DLIO_TFRECORD, FormatType.DALI_TFRECORD, FormatType.TF_TFRECORD]:
            from dlio_benchmark.data_generator.tf_generator import TFRecordGenerator
            return TFRecordGenerator()
        elif type == FormatType.DLIO_HDF5:
            from dlio_benchmark.data_generator.hdf5_generator import HDF5Generator
            return HDF5Generator()
        elif type == FormatType.DLIO_CSV:
            from dlio_benchmark.data_generator.csv_generator import CSVGenerator
            return CSVGenerator()
        elif type == FormatType.DLIO_NPZ:
            from dlio_benchmark.data_generator.npz_generator import NPZGenerator
            return NPZGenerator()
        elif type == FormatType.DALI_NPZ:
            from dlio_benchmark.data_generator.npy_generator import NPYGenerator
            return NPYGenerator()
        elif type == FormatType.DLIO_JPEG:
            from dlio_benchmark.data_generator.jpeg_generator import JPEGGenerator
            return JPEGGenerator()
        elif type == FormatType.DLIO_PNG:
            from dlio_benchmark.data_generator.png_generator import PNGGenerator
            return PNGGenerator()
        else:
            raise Exception(str(ErrorCodes.EC1001))