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

from src.common.enumerations import FrameworkType
from src.common.error_code import ErrorCodes


class FrameworkFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_framework(framework_type, profiling):
        if framework_type == FrameworkType.TENSORFLOW:
            from src.framework.tf_framework import TFFramework
            return TFFramework.get_instance(profiling)
        elif framework_type == FrameworkType.PYTORCH:
            from src.framework.torch_framework import TorchFramework
            return TorchFramework.get_instance(profiling)
        else:
            raise Exception(str(ErrorCodes.EC1001))