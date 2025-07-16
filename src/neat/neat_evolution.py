# src/neat/neat_evolution.py
import numpy as np
import random
from typing import List, Tuple
from .neat import NEATGenome
from .species import SpeciesManager
from .crossover import crossover_with_mutation

class NEATEvolution:    
    def __init__(self, config: dict):
        self.config = config
        self.species_manager = SpeciesManager(config)
    
    def create_next_generation(self, population: List[NEATGenome]) -> List[NEATGenome]:
        # Speciate the population
        self.species_manager.speciate(population)
        
        # Calculate offspring counts per species
        offspring_counts = self._calculate_offspring_counts()
        
        new_population = []
        
        # Elitism: keep best from each species
        for species in self.species_manager.species.values():
            if species.members and offspring_counts.get(species.id, 0) > 0:
                # Sort by fitness
                species.members.sort(key=lambda g: g.fitness or 0, reverse=True)
                # Keep the best
                new_population.append(species.members[0].copy())
        
        # Generate rest of offspring
        for species_id, count in offspring_counts.items():
            species = self.species_manager.species.get(species_id)
            if not species:
                continue

            remaining = count - 1 if new_population else count
            
            for _ in range(remaining):
                if len(species.members) == 1:
                    # Only one member, mutate it
                    parent = species.members[0]
                    child = parent.copy()
                    child.mutate(self.config)
                else:
                    # Genome/body mismatch issues - just use mutation                    
                    parent = self._tournament_select(species.members)
                    child = parent.copy()
                    child.mutate(self.config)
                
                new_population.append(child)
        
        # Remove stagnant species for next generation
        self.species_manager.cull_stagnant_species()
        
        return new_population
    
    def _calculate_offspring_counts(self) -> dict:
        # Get adjusted fitnesses
        species_fitness = {}
        for sid, species in self.species_manager.species.items():
            species_fitness[sid] = species.get_adjusted_fitness()
        
        total_fitness = sum(species_fitness.values())
        if total_fitness == 0:
            # Equal distribution if no fitness yet
            count = self.config['population_size'] // len(self.species_manager.species)
            return {sid: count for sid in self.species_manager.species}
        
        # Proportional to fitness
        offspring_counts = {}
        total_offspring = 0
        
        for sid, fitness in species_fitness.items():
            count = int((fitness / total_fitness) * self.config['population_size'])
            count = max(1, count)  # At least 1 offspring
            offspring_counts[sid] = count
            total_offspring += count
        
        # Adjust for rounding errors
        if total_offspring < self.config['population_size']:
            # Give extra to best species
            best_species = max(species_fitness, key=species_fitness.get)
            offspring_counts[best_species] += self.config['population_size'] - total_offspring
        
        return offspring_counts
    
    def _tournament_select(self, members: List[NEATGenome]) -> NEATGenome:
        tournament_size = self.config.get('tournament_size', 3)
        tournament_size = min(tournament_size, len(members))
        
        tournament = random.sample(members, tournament_size)
        return max(tournament, key=lambda g: g.fitness or 0)