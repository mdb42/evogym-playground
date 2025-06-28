# evogym-playground/src/simulation/evolution.py
"""
Evolution module for EvolutionGym robots.
"""

import numpy as np

def tournament_select(population, tournament_size):
    """Select an individual using tournament selection"""
    tournament = np.random.choice(population, size=tournament_size, replace=False)
    winner = max(tournament, key=lambda ind: ind.fitness)
    return winner.copy()  # Return a copy to avoid modifying original

def create_next_generation(population, config):
    """Create next generation (tournament selection)"""
    new_population = []
    
    # Sort by fitness
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    
    # Elitism - keep best N unchanged
    for i in range(min(config['elitism'], len(population))):
        new_population.append(population[i].copy())
    
    # Fill rest with mutated offspring
    while len(new_population) < config['population_size']:
        winner = tournament_select(population, config['tournament_size'])
        child = winner.mutate(config['mutation_rate'], config['mutation_amount'])
        new_population.append(child)
    
    return new_population
