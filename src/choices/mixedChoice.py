from ..choice import Choice
import numpy as np

class MixedChoice(Choice):
    def __init__(self, dir_choice : Choice, choices : list[Choice], epsilon = 1e-11):
        super().__init__()
        self.choice = dir_choice
        self.choices = choices

    def count_params(self):
        return self.choice.count_params() + sum([choice.count_params() for choice in self.choices])

    def __call__(self, n, params):
        params = self.apply_constraints(params)

        ci = self.choice(len(self.choices), params[:self.choice.count_params()])
        start = self.choice.count_params() + sum([c.count_params() for c in self.choices[:ci]]) if ci > 0 else self.choice.count_params()
        
        return self.choices[ci](n, params[start: start+self.choices[ci].count_params()])

    def apply_constraints(self, params):
        choice = self.choice.apply_constraints(params[:self.choice.count_params()])
        choices = []

        start = self.choice.count_params()

        for c in self.choices:
            end = start + c.count_params()
            choices.extend([param for param in c.apply_constraints(params[start:end])])
            start = end
        
        return choice + choices