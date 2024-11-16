from choice import Choice

class BigChoice(Choice):
    def __init__(self, epsilon = 1e-11):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, n, *params):
        return 0
    
    def apply_constraints(self, *params):
        return params