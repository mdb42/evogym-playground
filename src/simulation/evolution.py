
import numpy as np
import random


def is_connected(body):
    """Check if all non-empty voxels in body are connected"""
    if np.sum(body > 0) == 0:
        return False
    
    # Find all non-empty positions
    positions = np.argwhere(body > 0)
    if len(positions) == 0:
        return False
    
    # Flood fill from first non-empty position
    visited = set()
    to_visit = [tuple(positions[0])]
    
    while to_visit:
        pos = to_visit.pop()
        if pos in visited:
            continue
        visited.add(pos)
        
        # Check 4-connected neighbors
        i, j = pos
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < body.shape[0] and 0 <= nj < body.shape[1]:
                if body[ni, nj] > 0 and (ni, nj) not in visited:
                    to_visit.append((ni, nj))
    
    # Check if all non-empty positions were visited
    return len(visited) == len(positions)

def mutate_robot(body, connections, mutation_rate=0.1, mutation_amount=0.3):
    """Simple mutation of robot morphology"""
    body_copy = body.copy()
    
    # Try mutation up to 10 times (might be invalid)
    for attempt in range(10):
        temp_body = body_copy.copy()
        
        # Mutate some voxels
        for i in range(temp_body.shape[0]):
            for j in range(temp_body.shape[1]):
                if random.random() < mutation_rate:
                    current = temp_body[i, j]
                    if current == 0:  # Empty - maybe add something
                        if random.random() < mutation_amount:
                            temp_body[i, j] = random.choice([1, 2, 3, 4])
                    else:  # Non-empty - maybe change or remove
                        if random.random() < mutation_amount:
                            temp_body[i, j] = 0
                        else:
                            temp_body[i, j] = random.choice([1, 2, 3, 4])
        
        # Check if result is valid
        if is_connected(temp_body):
            return temp_body, None
    
    # If not a valid mutation, return original
    return body.copy(), None

def create_next_generation(population, fitnesses, config):
    """Create next generation (tournament selection)"""
    new_population = []
    
    # Elitism - keep best N unchanged
    sorted_indices = np.argsort(fitnesses)[::-1]  # Best to worst
    for i in range(config['elitism']):
        if i < len(population):
            new_population.append(population[sorted_indices[i]])
    
    # Fill rest with mutated offspring
    while len(new_population) < config['population_size']:
        # Tournament selection
        tournament_indices = random.sample(range(len(population)), config['tournament_size'])
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        
        # Mutate winner
        parent = population[winner_idx]
        child_body, child_connections = mutate_robot(
            parent[0], parent[1], 
            config['mutation_rate'], 
            config['mutation_amount']
        )
        new_population.append((child_body, child_connections))
    
    return new_population
