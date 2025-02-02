from abc import ABC

class Choice(ABC):
    def __init__(self):
        super().__init__()

    def count_params(self):
        pass

    def __call__(self, n, params):
        pass

    def apply_constraints(self, params):
        pass