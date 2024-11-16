from choice import Choice
import numpy as np
import random

class XLogChoice(Choice):
    def __init__(self, epsilon = 1e-11):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, n, *params):
        dx, a = params[0] if len(params) == 1 else params
        return int((dx / a) * (((a * n + dx) / dx) ** random.random() - 1))

    def apply_constraints(self, *params):
        dx, a = params
        return np.array([[max(dx, self.epsilon), max(a, self.epsilon)]])