from ..choiceParameters import ChoiceParameters
import numpy as np
from .pso import pso_solve

class PSOParameters(ChoiceParameters):
    params = None

    def __call__(self, embedding):
        return self.params
    
    def fit(self, fitness, constraints, number_of_params : int, number_of_particles : int, max_iterations : int):
        init_positions = lambda num_particles: [np.random.random((1, number_of_params)) * 2 - 1 for i in range(num_particles)]
        init_velocities = lambda num_particles: [np.zeros((1, number_of_params)) for i in range(num_particles)]   

        p, score, positions, velocities = pso_solve(number_of_particles, max_iterations
                                            , init_positions
                                            , init_velocities
                                            , fitness
                                            , lambda p: np.array([constraints(p[0])])
                                            , lambda score: False
                                            , False
                                            )
        
        self.params = p[0]
        return score, p[0]