# src/simulation/evolution.py
"""
Evolution module for EvolutionGym robots.
"""

import numpy as np
import random

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

    genomes = [ind.genome for ind in population]
    species_manager.speciate(genomes)
    
    remaining_pop_size = pop_size - len(new_population)
    offspring_counts = _calculate_offspring(species_manager, remaining_pop_size)
    
    # Helper for aggressive parent selection via tournament
    def tournament_select_genome(pool, size):
        # Ensure tournament size is not larger than the pool of candidates
        selection_size = min(len(pool), size)
        if selection_size == 0:
            return None
        selected = random.sample(pool, selection_size)
        return max(selected, key=lambda g: g.fitness)

    # Generate offspring
    while len(new_population) < pop_size:
        species = species_manager.select_species_for_reproduction()
        if not species or not species.members:
            # This can happen if all species have zero fitness, just pick one
            if species_manager.species:
                species = random.choice(list(species_manager.species.values()))
            else: # Catastrophic failure, repopulate randomly
                break 

        # Only the top 50% of a species can be parents
        species.members.sort(key=lambda g: g.fitness, reverse=True)
        cutoff = max(1, len(species.members) // 2)
        parent_pool = species.members[:cutoff]

        # Select using a tournament
        tournament_size = config.get('tournament_size', 3)

        if len(parent_pool) > 1 and random.random() < neat_config.get('crossover_rate', 0.75):
            # Crossover
            p1_genome = tournament_select_genome(parent_pool, tournament_size)
            
            # Decide between intra-species and inter-species crossover
            if random.random() < neat_config.get('interspecies_crossover_chance', 0.01):
                other_species = species_manager.select_species_for_reproduction()
                if other_species and other_species.members and other_species.id != species.id:
                    other_species.members.sort(key=lambda g: g.fitness, reverse=True)
                    other_cutoff = max(1, len(other_species.members) // 2)
                    other_parent_pool = other_species.members[:other_cutoff]
                    p2_genome = tournament_select_genome(other_parent_pool, tournament_size)
                else: # Fallback to intra-species
                    p2_genome = tournament_select_genome(parent_pool, tournament_size)
            else: # Normal intra-species crossover
                p2_genome = tournament_select_genome(parent_pool, tournament_size)

            if p1_genome is None or p2_genome is None: continue

            parent1 = genome_to_individual_map[p1_genome.key]
            parent2 = genome_to_individual_map[p2_genome.key]
            child = parent1.crossover(parent2)
        else:
            # Asexual reproduction (mutation only)
            p_genome = tournament_select_genome(parent_pool, tournament_size)
            if p_genome is None:
                continue
            
            parent = genome_to_individual_map[p_genome.key]
            child = parent.copy()
        
        # Always mutate the new child before adding it to pop
        mutated_child = child.mutate(
            config.get('mutation_rate', 0.1),
            config.get('mutation_amount', 0.3),
            neat_config=neat_config
        )
        new_population.append(mutated_child)
        
    # Failsafe to ensure population is the correct size
    while len(new_population) < pop_size:
        new_population.append(population[0].copy().mutate(
            config.get('mutation_rate', 0.1),
            config.get('mutation_amount', 0.3),
            neat_config=neat_config
        ))

    return new_population[:pop_size]


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
