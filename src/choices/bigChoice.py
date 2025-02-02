from ..choice import Choice

class BigChoice(Choice):
    def __init__(self):
        super().__init__()

    def count_params(self):
        return 0

    def __call__(self, n, params):
        return 0
    
    def apply_constraints(self, params):
        return params