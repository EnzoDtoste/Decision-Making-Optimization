from ..choice import Choice
import numpy as np

class ExpChoice(Choice):
    def __init__(self, epsilon = 1e-11):
        super().__init__()
        self.epsilon = epsilon

    def count_params(self):
        return 1

    def __call__(self, n, params):
        a_cons = self.apply_constraints(params)
        a = a_cons[0]

        if n is None:
            return int(np.random.exponential(a))
    
    def apply_constraints(self, params):
        a = params[0]
        return [max(a, self.epsilon)]