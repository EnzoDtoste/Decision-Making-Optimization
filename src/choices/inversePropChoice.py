from ..choice import Choice
import numpy as np

class InversePropChoice(Choice):
    def __init__(self, epsilon = 1e-11):
        super().__init__()
        self.epsilon = epsilon

    def count_params(self):
        return 2

    def __call__(self, n, params):
        a, b = self.apply_constraints(params)

        if n is None:
            r = np.random.random()
            return int((a * r) / (b * (1 - r)))
    
    def apply_constraints(self, params):
        a, b = params
        return [max(a, 0), max(b, self.epsilon)]