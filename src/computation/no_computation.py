from src.computation.computation_handler import ComputationHandler


class NoComputation(ComputationHandler):
    def __init__(self):
        super().__init__()

    def compute(self):
        pass