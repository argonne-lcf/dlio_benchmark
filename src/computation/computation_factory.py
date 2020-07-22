from src.common.enumerations import ComputationType
from src.common.error_code import ErrorCodes
from src.computation.asynchronous_computation import AsyncComputation
from src.computation.no_computation import NoComputation
from src.computation.synchronous_computation import SyncComputation


class ComputationFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_handler(type):
        if type == ComputationType.NONE:
            return NoComputation()
        elif type == ComputationType.ASYNC:
            return AsyncComputation()
        elif type == ComputationType.SYNC:
            return SyncComputation()
        else:
            raise Exception(str(ErrorCodes.EC1000))