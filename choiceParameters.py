from abc import ABC

class ChoiceParameters(ABC):
    def reset_state(self):
        pass

    def __call__(self, embedding):
        pass