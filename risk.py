from problem import Problem
from choice import Choice
import random
import math

class RiskGame(Problem):
    def __init__(self, choice, choiceParameters, rounds):
        super().__init__(choice, choiceParameters)
        self.rounds = rounds

    def get_current_embedding(self, *params):
        wol = params
        return wol

    def select_choice(self, *params):
        embedding = self.get_current_embedding(*params)
        self.embeddings.append(embedding)
        return self.choice(None, self.choiceParameters(embedding))
    
    def run(self, get_prob_risk):
        self.reset_embeddings()
        self.choiceParameters.reset_state()

        acc = 0
        wol = [0 for _ in range(self.rounds + 1)]
        
        for i in range(self.rounds):
            wol[0] = acc
            n = self.select_choice(wol)
            wol[i + 1] = -1 if random.random() < get_prob_risk(i) else 1
            acc += wol[i + 1] * n

        return acc
