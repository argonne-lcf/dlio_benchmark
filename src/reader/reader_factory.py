"""
   Copyright 2021 UChicago Argonne, LLC

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
    def get_reader(type, data_loader):
        """
        This function set the data reader based on the data format and the data loader specified. 
        """
        if data_loader==None:
            if type == FormatType.HDF5:
                from src.reader.hdf5_reader import HDF5Reader
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
            else:
                print("Loading data of %s format is not supported without framework data loader" %type)
                raise Exception(str(ErrorCodes.EC1004))
        elif data_loader == DataLoaderType.TENSORFLOW:
            if type == FormatType.TFRECORD:
                from src.reader.tf_reader import TFReader
                return TFReader()
            from src.reader.tf_data_loader_reader import TFDataLoaderReader
            return TFDataLoaderReader()
        elif data_loader == DataLoaderType.PYTORCH:
            from src.reader.torch_data_loader_reader import TorchDataLoaderReader
            return TorchDataLoaderReader()
        # Implement other data loader here
        else:
            print("Data Loader %s is not implemented" %data_loader)
            raise Exception(str(ErrorCodes.EC1004))