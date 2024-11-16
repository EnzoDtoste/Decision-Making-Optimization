import numpy as np

# PSO
def pso_solve(num_particles, max_iterations, init_positions, init_velocities, fitness, apply_constraints, stop, enable_reset = True):
    # Parámetros de PSO
    w = 0.5  # Inercia
    cp1 = 1.5  # Cognición (partícula individual)
    cp2 = 1.5 # Social (mejor global)

    # Inicialización
    size_pos = num_particles

    positions = init_positions(size_pos)
    velocities = init_velocities(size_pos)
    
    personal_best_positions = np.copy(positions)
    personal_best_scores = np.full(size_pos, None)

    global_best_score = None
    global_best_position = None

    first_iteration = True
    fitnesses = []

    count_reset = 0

    global_global_best_score = None
    global_global_best_position = None

    # Bucle principal de PSO
    for iteration in range(max_iterations):
        for i in range(size_pos):
            # Se ajusta las posiciones, para que siempre sean válidas con respecto a las restricciones
            positions[i] = apply_constraints(positions[i])

            # Se obtiene la evaluación
            score = fitness(positions[i])
            
            if first_iteration:
                fitnesses.append((i, score))

            # Que tan cerca está la partícula de la solución global encontrada hasta ahora
            if enable_reset and global_best_position is not None and all([abs(e) < 0.2 for e in (global_best_position[0] - positions[i][0])]):
                count_reset += 1

            # Actualizar el mejor personal
            if personal_best_scores[i] is None or score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]
                
                # Actualizar el mejor global
                if global_best_score is None or score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                    if global_global_best_score is None or global_best_score < global_global_best_score:
                        global_global_best_score = global_best_score
                        global_global_best_position = global_best_position

                    # Si la evaluación es lo suficientemente buena se detiene el algoritmo
                    if stop(global_global_best_score):
                        return global_global_best_position, global_global_best_score, positions, velocities

            # Actualizar velocidad
            velocities[i] = w * velocities[i] + \
                            cp1 * np.random.rand() * (personal_best_positions[i] - positions[i]) + \
                            cp2 * np.random.rand() * (global_best_position - positions[i])
                            
            # Actualizar posición
            positions[i] = positions[i] + velocities[i]

        
        # Si menos del 60% de las partículas está en areas distintas a la mejor encontrada
        if count_reset < 2 * size_pos / 3:
            if first_iteration:
                fitnesses = sorted(fitnesses, key=lambda f: f[1])[:num_particles]
                
                positions = [positions[i] for i, score in fitnesses]
                velocities = [velocities[i] for i, score in fitnesses]
                personal_best_positions = [personal_best_positions[i] for i, score in fitnesses]
                personal_best_scores = [personal_best_scores[i] for i, score in fitnesses]

                first_iteration = False
                size_pos = num_particles

        # Si más del 60% de las partículas están alrededor del mínimo global
        else:
            size_pos = num_particles
            positions = init_positions(size_pos)
            velocities = init_velocities(size_pos)
            
            personal_best_positions = np.copy(positions)
            personal_best_scores = np.full(size_pos, np.inf)

            global_best_score = np.inf
            global_best_position = None

            first_iteration = True
            fitnesses = []
            #print("RESET")
        
        #print("COUNT" + str(count_reset))
        count_reset = 0

    # Se devuelve el mejor encontrado
    return global_global_best_position, global_global_best_score, positions, velocities