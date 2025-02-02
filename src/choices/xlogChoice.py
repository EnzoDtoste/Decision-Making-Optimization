from ..choice import Choice
import numpy as np

class XLogChoice(Choice):
    def __init__(self, epsilon = 1e-11):
        super().__init__()
        self.epsilon = epsilon

    def count_params(self):
        return 2

    def __call__(self, n, params):
        dx, a = self.apply_constraints(params)
        return min(int((dx / a) * (((a * n + dx) / dx) ** np.random.random() - 1)), n - 1)

    def apply_constraints(self, params):
        dx, a = params
        return [max(dx, self.epsilon), max(a, self.epsilon)]