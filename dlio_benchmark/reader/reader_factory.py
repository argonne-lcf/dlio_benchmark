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

from dlio_benchmark.common.enumerations import FormatType, DataLoaderType
from dlio_benchmark.common.error_code import ErrorCodes


class ReaderFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_reader(type, dataset_type, thread_index, epoch_number):
        """
        This function set the data reader based on the data format and the data loader specified. 
        """

        if type == FormatType.HDF5:
            from dlio_benchmark.reader.hdf5_reader import HDF5Reader
            return HDF5Reader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.CSV:
            from dlio_benchmark.reader.csv_reader import CSVReader
            return CSVReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.JPEG:
            from dlio_benchmark.reader.jpeg_reader import JPEGReader
            return JPEGReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.PNG:
            from dlio_benchmark.reader.png_reader import PNGReader
            return PNGReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.NPZ:
            from dlio_benchmark.reader.npz_reader import NPZReader
            return NPZReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.TFRECORD:
            from dlio_benchmark.reader.tf_reader import TFReader
            return TFReader(dataset_type, thread_index, epoch_number)
        else:
            print("Loading data of %s format is not supported without framework data loader" %type)
            raise Exception(type)
