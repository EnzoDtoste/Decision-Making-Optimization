from problem import Problem
from choice import Choice
import random
import math
import numpy as np
from sortedcontainers import SortedDict

class MemoryGame(Problem):
    def __init__(self, choice : Choice, choiceParameters, prob_forget, N, limit):
        super().__init__(choice, choiceParameters)
        self.prob_forget = prob_forget
        self.N = N
        self.limit = limit
        self.cols = math.gcd(int(math.sqrt(N)), N)
        self.rows = N // self.cols

    def get_current_embedding(self, *params):
        B, _ = params
        return B

    def order_choices(self, choices, *params):
        B, _ = params
        sorted_dict = SortedDict()
        
        for i, j in choices:
            sorted_dict[np.dot(B[i], B[j])] = (i, j)

        choices = list(sorted_dict.values())
        choices.reverse()
        return choices

    def get_choices(self, *params):
        B, discovered = params
        N = len(B)
        plays = []
        for i in range(N):
            if not discovered[i]:
                for j in range(i+1, N):
                    if not discovered[j]:
                        plays.append((i, j))

        return plays

    def obtain_normal_values_truncated(self, N, num_values):
        mu = self.prob_forget * N
        sigma = (self.prob_forget / 2) * N
        #a = (0 - mu) / sigma 
        #b = 10
        #a, b = (0 - mu) / sigma, (N - mu) / sigma
        #valores = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=num_values)
        valores = np.random.normal(mu, sigma, num_values)
        valores = [max(0, min(v, N)) for v in valores]
        suma_valores = sum(valores)
        
        if suma_valores > 0:
            factor_normalizacion = N / suma_valores if suma_valores > N else 1
            valores_normalizados = [v * factor_normalizacion for v in valores]
            return [min(v, N) for v in valores_normalizados]
        
        return valores

    def limitacion(self, B, discovered):
        N, num_cartas = B.shape
        newB = np.zeros((N, num_cartas))

        for i in range(N):
            if not discovered[i]:
                fila = i // self.rows
                columna = i % self.cols

                for j in range(num_cartas):
                    prob_actual = B[i][j]
                    discount = 0

                    if prob_actual > 0:
                        vecinos = []

                        for delta_fila in [-1, 0, 1]:
                            for delta_columna in [-1, 0, 1]:
                                if delta_fila == 0 and delta_columna == 0:
                                    continue
                                vecino_fila = fila + delta_fila
                                vecino_columna = columna + delta_columna

                                if 0 <= vecino_fila < self.rows and 0 <= vecino_columna < self.cols:
                                    vecino_indice = vecino_fila * self.cols + vecino_columna
                                    if not discovered[vecino_indice]:
                                        vecinos.append(vecino_indice)

                        num_vecinos = len(vecinos)
                        if num_vecinos > 0:
                            probs = self.obtain_normal_values_truncated(prob_actual, num_vecinos)
                            discount = sum(probs)

                            for vecino_indice, prob in zip(vecinos, probs):
                                newB[vecino_indice][j] += prob

                    newB[i][j] += prob_actual - discount

        return newB

    def random_cards(self):
        num_cards = self.N // 2
        cards = list(range(num_cards)) * 2
        random.shuffle(cards)
        return cards

    def run(self, cards):
        self.reset_embeddings()
        self.choiceParameters.reset_state()

        num_cards = len(cards) // 2
        discovered = [False] * self.N

        B = np.ones((self.N, num_cards)) / self.N

        moves = 0
        while not all(discovered):
            B = self.limitacion(B, discovered)

            i, j = self.select_choice(B, discovered)

            c_i = cards[i]
            c_j = cards[j]
            
            if c_i == c_j:
                discovered[i] = True
                discovered[j] = True

                B[i, :] = 0
                B[j, :] = 0

                B[:, c_i] = 0
                B[:, c_j] = 0
            else:
                B[i, :] = 0
                B[i, c_i] = 1
                B[j, :] = 0
                B[j, c_j] = 1

            moves += 1

            if moves >= self.limit:
                break
            
            for k in range(num_cards):
                total = sum(B[:, k])
                if total > 0:
                    B[:, k] /= total
            
        
        return moves