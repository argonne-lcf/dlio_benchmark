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
import logging
from dlio_benchmark.utils.utility import utcnow

from dlio_benchmark.utils.config import ConfigArguments

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

        _args = ConfigArguments.get_instance()
        if _args.reader_class is not None:
            logging.info(f"{utcnow()} Running DLIO with custom data loader class {_args.reader_class.__name__}")
            return _args.reader_class(dataset_type, thread_index, epoch_number)
        elif type == FormatType.DLIO_HDF5:
            from dlio_benchmark.reader.dlio_hdf5_reader import DLIOHDF5Reader
            return DLIOHDF5Reader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.DLIO_CSV:
            from dlio_benchmark.reader.dlio_csv_reader import DLIOCSVReader
            return DLIOCSVReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.DLIO_JPEG:
            from dlio_benchmark.reader.dlio_jpeg_reader import DLIOJPEGReader
            return DLIOJPEGReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.DLIO_PNG:
            from dlio_benchmark.reader.dlio_png_reader import DLIOPNGReader
            return DLIOPNGReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.DLIO_NPZ:
            from dlio_benchmark.reader.dlio_npz_reader import DLIONPZReader
            return DLIONPZReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.DLIO_TFRECORD:
            from dlio_benchmark.reader.dlio_tfrecord_reader import DLIOTFRecordReader
            return DLIOTFRecordReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.TF_TFRECORD:
            from dlio_benchmark.reader.tf_tfrecord_reader import TFTFRecordReader
            return TFTFRecordReader(dataset_type)
        elif type == FormatType.DALI_TFRECORD:
            from dlio_benchmark.reader.dali_tfrecord_reader import DaliTFRecordReader
            return DaliTFRecordReader(dataset_type)
        elif type == FormatType.DALI_NPZ:
            from dlio_benchmark.reader.dali_npz_reader import DaliNPZReader
            return DaliNPZReader(dataset_type)
        else:
            print("Loading data of %s format is not supported without framework data loader" %type)
            raise Exception(type)
