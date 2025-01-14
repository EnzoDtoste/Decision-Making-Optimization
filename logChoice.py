from choice import Choice, Size
import numpy as np

class LogChoice(Choice):
    def __init__(self, epsilon = 1e-11, size = Size.ALWAYS_FINITE):
        super().__init__(size)
        self.epsilon = epsilon

    def count_params(self):
        return 2

    def __call__(self, n, *params):
        a, b = params[0] if len(params) == 1 else params
        ab_cons = self.apply_constraints(a, b)
        a, b = ab_cons[0][0], ab_cons[0][1]

        if n is not None:
            exp = a ** (1 / b)
            if exp == np.inf:
                return n - 1 if (1 - np.random.random()) < self.epsilon else 0
            if abs(exp - 1) < self.epsilon:
                return n - 1
            return min(int(n * (exp ** np.random.random() - 1) / (exp - 1)), n - 1)
        else:
            r = np.random.random()
            return int(a / (b * (r - 1)) + a / b)
    
    def adjust_zero(self, value):
        if abs(value) < self.epsilon:
            return self.epsilon if value >= 0 else -self.epsilon
        return value

    def apply_constraints(self, *params):
        a, b = params

        if self.size == Size.ALWAYS_FINITE:
            return np.array([[max(a, self.epsilon), self.adjust_zero(b)]])
        elif self.size == Size.ALWAYS_INFINITE:
            return np.array([[max(a, 0), min(b, -self.epsilon)]])
        else:
            return np.array([[max(a, self.epsilon), min(b, -self.epsilon)]])