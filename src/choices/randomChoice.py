from ..choice import Choice
import random

class RandomChoice(Choice):
    def __init__(self):
        super().__init__()

    def count_params(self):
        return 0

    def __call__(self, n, params):
        if n is not None:
            return random.randint(0, n - 1)
        
    def apply_constraints(self, params):
        return params