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