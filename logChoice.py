from choice import Choice
import numpy as np

class LogChoice(Choice):
    def __init__(self, epsilon = 1e-11):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, n, *params):
        x, b = params[0] if len(params) == 1 else params
        exp = x ** (1 / b)
        if exp == np.inf:
            return n - 1 if (1 - np.random.random()) < self.epsilon else 0
        return min(int(n * (exp ** np.random.random() - 1) / (exp - 1)), n - 1)
    
    def adjust_zero(self, value):
        if abs(value) < self.epsilon:
            return self.epsilon if value >= 0 else -self.epsilon
        return value

    def apply_constraints(self, *params):
        x, b = params
        return np.array([[max(x, self.epsilon), self.adjust_zero(b)]])