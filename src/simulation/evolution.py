# src/simulation/evolution.py
"""
Evolution module for EvolutionGym robots.
"""

import numpy as np
import random
from src.individual.neat_individual import NEATIndividual
from src.neat.species import SpeciesManager

def tournament_select(population, tournament_size):
    """Select an individual using tournament selection"""
    tournament = np.random.choice(population, size=tournament_size, replace=False)
    winner = max(tournament, key=lambda ind: ind.fitness)
    return winner.copy()  # Return a copy to avoid modifying original

def create_next_generation(population, species_manager, config):
    control_type = config.get('control_type', 'random')
    
    if control_type == 'neat':
        return _create_neat_next_generation(population, species_manager, config)
    else:
        return _create_random_next_generation(population, config)

def _create_neat_next_generation(population, species_manager, config):
    neat_config = config.get('neat_config', {})
    pop_size = config['population_size']
    new_population = []

    # Sort by fitness
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    num_global_elites = config.get('elitism', 1)

    # Elitism - keep the best individuals
    for i in range(min(num_global_elites, len(population))):
        new_population.append(population[i].copy())

    # Create an efficient map to look up an Individual from its Genome's key
    genome_to_individual_map = {ind.genome.key: ind for ind in population}

    # Sync fitness from each Individual to its corresponding Genome
    for ind in population:
        ind.genome.fitness = ind.fitness if ind.fitness is not None else -float('inf')

    # Divide the population into species
    genomes = [ind.genome for ind in population]
    species_manager.speciate(genomes)
    
    # Calculate how many offspring each species should produce for the remaining slots
    remaining_pop_size = pop_size - len(new_population)
    offspring_counts = _calculate_offspring(species_manager, remaining_pop_size)
    
    # Generate offspring    
    while len(new_population) < pop_size:
        # Select a species to reproduce from
        species = species_manager.select_species_for_reproduction()
        if not species or not species.members:
            continue

        # Only the top 50% of a species can be parents
        species.members.sort(key=lambda g: g.fitness, reverse=True)
        cutoff = max(1, len(species.members) // 2)
        parent_pool = species.members[:cutoff]

        # Select from the high-performing pool
        if len(parent_pool) > 1 and random.random() < neat_config.get('crossover_rate', 0.75):
            # Crossover
            p1_genome = species_manager.select_member(parent_pool)
            p2_genome = species_manager.select_member(parent_pool)
            parent1 = genome_to_individual_map[p1_genome.key]
            parent2 = genome_to_individual_map[p2_genome.key]
            child = parent1.crossover(parent2)
        else:
            # Asexual reproduction (mutation only)
            p_genome = species_manager.select_member(parent_pool)
            parent = genome_to_individual_map[p_genome.key]
            child = parent.copy()
        
        # Always mutate the new child before adding it to pop
        mutated_child = child.mutate(
            config.get('mutation_rate', 0.1),
            config.get('mutation_amount', 0.3),
            neat_config=neat_config
        )
        new_population.append(mutated_child)

    return new_population


def _calculate_offspring(species_manager, num_to_spawn):
    species_fitness = {}
    for sid, species in species_manager.species.items():
        if species.members:
            species_fitness[sid] = species.get_adjusted_fitness()
    
    total_adj_fitness = sum(species_fitness.values())
    if total_adj_fitness <= 0:
        # Fallback to equal distribution
        if not species_fitness: return {}
        equal_share = num_to_spawn // len(species_fitness)
        return {sid: equal_share for sid in species_fitness}
    
    # Calculate offspring proportionally
    proportions = []
    for sid, fitness in species_fitness.items():
        proportion = (fitness / total_adj_fitness) * num_to_spawn
        proportions.append({'id': sid, 'prop': proportion})
    
    offspring_counts = {p['id']: int(p['prop']) for p in proportions}
    
    # Distribute remaining slots
    allocated = sum(offspring_counts.values())
    remainder = num_to_spawn - allocated
    
    proportions.sort(key=lambda p: p['prop'] - int(p['prop']), reverse=True)
    
    for i in range(remainder):
        offspring_counts[proportions[i]['id']] += 1
        
    return offspring_counts


def _create_random_next_generation(population, config):
    new_population = []
    
    # Sort by fitness
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    
    # Elitism
    for i in range(min(config.get('elitism', 2), len(population))):
        new_population.append(population[i].copy())
    
    # Fill rest with mutated offspring
    while len(new_population) < config['population_size']:
        winner = tournament_select(population, config.get('tournament_size', 3))
        child = winner.mutate(config['mutation_rate'], config['mutation_amount'])
        new_population.append(child)
    
    return new_population
