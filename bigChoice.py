from choice import Choice, Size

class BigChoice(Choice):
    def __init__(self, size = Size.ALWAYS_FINITE):
        super().__init__(size)

    def count_params(self):
        return 0

    def __call__(self, n, *params):
        return 0
    
    def apply_constraints(self, *params):
        return params