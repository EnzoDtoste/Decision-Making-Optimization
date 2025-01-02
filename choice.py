from abc import ABC
from enum import Enum

class Size(Enum):
    ALWAYS_FINITE = 1
    ALWAYS_INFINITE = 2
    MIXED = 3

class Choice(ABC):
    def __init__(self, size : Size):
        super().__init__()
        self.size = size

    def count_params(self):
        pass

    def __call__(self, n, *params):
        pass

    def apply_constraints(self, *params):
        pass