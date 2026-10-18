from abc import ABC
from .choice import Choice
from .choiceParameters import ChoiceParameters
from typing_extensions import TypeVar, Generic

TChoiceType = TypeVar('TChoiceType')

class Problem(ABC, Generic[TChoiceType]):
    def __init__(self, choice: Choice, choiceParameters : ChoiceParameters):
        super().__init__()
        self.choice = choice
        self.choiceParameters = choiceParameters
        self.embeddings = []

    def reset_embeddings(self):
        self.embeddings = []

    def get_current_embedding(self, params):
        ...

    def select_choice(self, params) -> TChoiceType:
        choices = self.get_choices(params)
        if len(choices) == 0:
            raise ValueError("No choices available")

        choices = self.order_choices(choices, params)
        embedding = self.get_current_embedding(params)
        self.embeddings.append(embedding)
        return choices[self.choice(len(choices), self.choiceParameters(embedding))]

    def order_choices(self, choices, params) -> list[TChoiceType]:
        ...

    def get_choices(self, params) -> list[TChoiceType]:
        ...

    def run(self, params):
        ...