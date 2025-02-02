from ..choice import Choice
import numpy as np

class LogChoice(Choice):
    def __init__(self, epsilon = 1e-11):
        super().__init__()
        self.epsilon = epsilon

    def count_params(self):
        return 2

    def __call__(self, n, params):
        a, b = self.apply_constraints(params)

        if n is not None:
            exp = a ** (1 / b)
            if exp == np.inf:
                return n - 1 if (1 - np.random.random()) < self.epsilon else 0
            if abs(exp - 1) < self.epsilon:
                return n - 1
            
            v = n * (exp ** np.random.random() - 1) / (exp - 1)

            if v != np.inf:
                return min(int(v), n - 1)
            else:
                return n - 1
    
    def adjust_zero(self, value):
        if abs(value) < self.epsilon:
            return self.epsilon if value >= 0 else -self.epsilon
        return value

    def apply_constraints(self, params):
        a, b = params
        return [max(a, self.epsilon), self.adjust_zero(b)]