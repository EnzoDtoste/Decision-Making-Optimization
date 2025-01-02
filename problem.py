from abc import ABC
from choice import Choice
from choiceParameters import ChoiceParameters

class Problem(ABC):
    def __init__(self, choice, choiceParameters : ChoiceParameters):
        super().__init__()
        self.choice : Choice = choice
        self.choiceParameters = choiceParameters
        self.embeddings = []

    def reset_embeddings(self):
        self.embeddings = []

    def get_current_embedding(self, *params):
        pass

    def select_choice(self, *params):
        choices = self.get_choices(*params)
        if len(choices) == 0:
            return

        choices = self.order_choices(choices, *params)
        embedding = self.get_current_embedding(*params)
        self.embeddings.append(embedding)
        return choices[self.choice(len(choices), self.choiceParameters(embedding))]

    def order_choices(self, choices, *params) -> list:
        pass

    def get_choices(self, *params) -> list:
        pass

    def run(self, *params):
        pass