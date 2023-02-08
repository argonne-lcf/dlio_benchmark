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


class DataLoaderFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_loader(type, format_type, dataset_type):
        """
        This function set the data reader based on the data format and the data loader specified.
        """

        if type == DataLoaderType.PYTORCH:
            from src.data_loader.torch_data_loader import TorchDataLoader
            return TorchDataLoader(format_type, dataset_type)
        elif type == DataLoaderType.TENSORFLOW:
            from src.data_loader.tf_data_loader import TFDataLoader
            return TFDataLoader(format_type, dataset_type)
        else:
            print("Data Loader %s not supported" %type)
            raise Exception(str(ErrorCodes.EC1004))
