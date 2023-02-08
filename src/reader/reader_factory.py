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

from src.common.enumerations import FormatType, DataLoaderType
from src.common.error_code import ErrorCodes


class ReaderFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_reader(type, dataset_type):
        """
        This function set the data reader based on the data format and the data loader specified. 
        """

        if type == FormatType.HDF5:
            from src.reader.hdf5_reader import HDF5Reader
            return HDF5Reader(dataset_type)
        elif type == FormatType.CSV:
            from src.reader.csv_reader import CSVReader
            return CSVReader(dataset_type)
        elif type == FormatType.JPEG:
            from src.reader.jpeg_reader import JPEGReader
            return JPEGReader(dataset_type)
        elif type == FormatType.PNG:
            from src.reader.png_reader import PNGReader
            return PNGReader(dataset_type)
        elif type == FormatType.NPZ:
            from src.reader.npz_reader import NPZReader
            return NPZReader(dataset_type)
        elif type == FormatType.TFRECORD:
            from src.reader.tf_reader import TFReader
            return TFReader(dataset_type)
        else:
            print("Loading data of %s format is not supported without framework data loader" %type)
            raise Exception(type)
