# src/simulation/scheduler.py
import numpy as np
import copy
from src.simulation.individual import NEATIndividual
from evogym import sample_robot

class Scheduler:
    """An active scheduler that observes the simulation and dynamically
    adjusts parameters to guide the evolutionary process."""
    def __init__(self, config, logger):
        self.logger = logger
        self.baseline_config = copy.deepcopy(config)
        self.scheduler_config = self.baseline_config.get('scheduler_config', {})
        self.best_fitness_history = []
        self.generations_without_improvement = 0
        self.zero_fitness_generations = 0

    def check_and_apply(self, generation, population, species_manager):
        """Called each generation to get a dynamically adjusted config."""
        current_config = copy.deepcopy(self.baseline_config)
        neat_config = current_config.get('neat_config', {})

        self._track_progress(population)

        # Check for critical state: is the entire population lifeless?
        if self._handle_lifeless_population(population, neat_config, current_config):
            # If so, take emergency measures and skip all other scheduling
            pass
        elif self._handle_stagnation(neat_config, current_config):
            # If stagnant, intervene
            pass
        else:
            # If stable, apply standard progressive scheduling
            self._apply_progressive_scheduling(generation, neat_config, current_config)
        
        # Always fine-tune the number of species
        self._dynamic_compatibility_adjustment(species_manager, neat_config)

        current_config['neat_config'] = neat_config
        return current_config

    def _track_progress(self, population):
        """Track the best fitness and update stagnation counters."""
        best_fitness = max([ind.fitness for ind in population if ind.fitness is not None] or [-float('inf')])
        if self.best_fitness_history and best_fitness > max(self.best_fitness_history) + 0.01:
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        self.best_fitness_history.append(best_fitness)

    def _handle_lifeless_population(self, population, neat_config, current_config):
        """Emergency intervention if the population shows no signs of life."""
        fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
        # Use a small threshold to account for minor sliding fitness
        if fitnesses and max(fitnesses) < 0.1:
            self.zero_fitness_generations += 1
            if self.zero_fitness_generations > self.scheduler_config.get('lifeless_limit', 20):
                self.logger.warning(f"SCHEDULER: Lifeless population for {self.zero_fitness_generations} gens! Forcing connection growth.")
                # Drastically increase the chance of adding connections
                neat_config['connection_add_rate'] = 0.9
                neat_config['weight_mutation_power'] = 1.5
                return True # Intervention applied
        else:
            self.zero_fitness_generations = 0
        return False

    def _handle_stagnation(self, neat_config, current_config):
        """Patient intervention for long-term plateaus."""
        stagnation_limit = self.scheduler_config.get('global_stagnation_limit', 50)
        if self.generations_without_improvement > stagnation_limit:
            self.logger.warning(f"SCHEDULER: Global stagnation for {self.generations_without_improvement} gens. Applying exploration burst.")
            neat_config['weight_mutation_power'] *= 1.5
            neat_config['connection_add_rate'] *= 1.5
            current_config['mutation_rate'] *= 1.2
            self.generations_without_improvement = 0 # Reset counter
            return True
        return False

    def _apply_progressive_scheduling(self, generation, neat_config, current_config):
        """A schedule optimized for building complexity from scratch."""
        if not self.scheduler_config.get('enable_parameter_scheduling', False): 
            return

        if generation < 100:  # Phase 1: Structure Building
            current_config['mutation_rate'] = 0.2  # Higher for exploration
            neat_config['connection_add_rate'] = 0.5
            neat_config['node_add_rate'] = 0.05
            neat_config['weight_mutation_power'] = 0.5  # Add this
            neat_config['weight_mutation_rate'] = 0.6   # Add this
            
        elif generation < 300:  # Phase 2: Complexification & Optimization
            current_config['mutation_rate'] = 0.15
            neat_config['connection_add_rate'] = 0.2
            neat_config['node_add_rate'] = 0.1
            neat_config['weight_mutation_power'] = 0.4
            neat_config['weight_mutation_rate'] = 0.8
            
        else:  # Phase 3: Fine-tuning
            current_config['mutation_rate'] = 0.05
            neat_config['connection_add_rate'] = 0.05
            neat_config['node_add_rate'] = 0.02
            neat_config['weight_mutation_power'] = 0.2
            neat_config['weight_mutation_rate'] = 0.95

    def _dynamic_compatibility_adjustment(self, species_manager, neat_config):
        """Adjusts speciation threshold to maintain a target number of niches."""
        if not species_manager: return
            
        current_count = len(species_manager.species)
        target = neat_config.get('target_species', 6)
        
        # More aggressive adjustment when starting from scratch
        if current_count < target:
            neat_config['compatibility_threshold'] *= 0.9
        elif current_count > target:
            neat_config['compatibility_threshold'] *= 1.1
        
        neat_config['compatibility_threshold'] = np.clip(
            neat_config['compatibility_threshold'], 1.5, 10.0
        )