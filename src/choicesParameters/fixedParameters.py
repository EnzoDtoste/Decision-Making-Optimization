from ..choiceParameters import ChoiceParameters

class FixedParameters(ChoiceParameters):
    params = None

    def set(self, params):
        self.params = params

    def __call__(self, embedding):
        return self.params