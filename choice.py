from abc import ABC

class Choice(ABC):
    def __call__(self, n, *params):
        pass

    def apply_constraints(self, *params):
        pass