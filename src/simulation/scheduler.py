# src/simulation/scheduler.py
"""
Scheduler module for dynamically adjusting evolutionary parameters
to guide the evolutionary process in EvolutionGym.
"""
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

    def check_and_apply(self, generation, population, species_manager):
        """Check the current state of the simulation and apply dynamic scheduling
        interventions based on the current generation and population state."""
        current_config = copy.deepcopy(self.baseline_config)
        neat_config = current_config.get('neat_config', {})

        # Track Progress
        self._track_progress(population)

        # Check and respond to specific evolutionary states
        if self._handle_stagnation(population, neat_config, current_config):
            pass # Stagnation intervention applied
        elif self._handle_convergence(population, species_manager, neat_config, current_config):
            pass # Convergence intervention applied
        else:
            # If stable, apply standard progressive scheduling
            self._apply_progressive_scheduling(generation, neat_config, current_config)
            self._apply_pressure_waves(generation, neat_config)
        
        # Always fine-tune compatibility
        self._dynamic_compatibility_adjustment(species_manager, neat_config)

        # Return the final config for this generation
        current_config['neat_config'] = neat_config
        return current_config

    def _track_progress(self, population):
        """Track the best fitness and update history."""
        best_fitness = max([ind.fitness for ind in population if ind.fitness is not None] or [-float('inf')])
        if self.best_fitness_history and best_fitness > max(self.best_fitness_history):
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        self.best_fitness_history.append(best_fitness)

    def _handle_stagnation(self, population, neat_config, current_config):
        """Escalating interventions based on stagnation duration"""
        if self.generations_without_improvement > 30:
            # EMERGENCY!
            self.logger.warning(f"SCHEDULER: EMERGENCY! {self.generations_without_improvement} gens without improvement!")
            neat_config['weight_mutation_power'] = 1.0
            neat_config['connection_add_rate'] = 0.5
            current_config['mutation_rate'] = 0.3
            neat_config['compatibility_threshold'] *= 0.5
            self.generations_without_improvement = 0
            return True
        elif self.generations_without_improvement > 15:
            # Standard intervention
            self.logger.warning(f"SCHEDULER: Stagnation for {self.generations_without_improvement} gens.")
            neat_config['weight_mutation_power'] *= 1.5
            neat_config['connection_add_rate'] *= 1.5
            neat_config['node_add_rate'] *= 1.5
            current_config['mutation_rate'] *= 1.5
            self.generations_without_improvement = 0
            return True
        return False

    def _handle_convergence(self, population, species_manager, neat_config, current_config):
        """If population has converged prematurely, force diversification."""
        if not species_manager or len(self.best_fitness_history) < 20: return False
        
        # Condition: Very few species and low fitness variance
        if len(species_manager.species) <= 2:
            fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
            if fitnesses and np.var(fitnesses) < 0.1:
                self.logger.warning("SCHEDULER: Premature convergence detected! Forcing diversity.")
                neat_config['compatibility_threshold'] *= 0.7 # Force new species
                current_config['mutation_rate'] = 0.25
                self._inject_random_individuals(population, current_config, percent=0.2)
                return True # Intervention was applied
        return False

    def _apply_progressive_scheduling(self, generation, neat_config, current_config):
        """Apply standard parameter schedules for different phases of the run."""
        if not self.scheduler_config.get('enable_parameter_scheduling', False): return

        if generation < 50: # Exploration Phase
            current_config['mutation_rate'] = 0.15
            neat_config['weight_mutation_power'] = 0.5
            neat_config['connection_add_rate'] = 0.3
            neat_config['node_add_rate'] = 0.1
        elif generation < 150: # Optimization Phase
            current_config['mutation_rate'] = 0.1
            neat_config['weight_mutation_power'] = 0.25
            neat_config['connection_add_rate'] = 0.15
            neat_config['node_add_rate'] = 0.05
        else: # Fine-tuning Phase
            current_config['mutation_rate'] = 0.05
            neat_config['weight_mutation_power'] = 0.15
            neat_config['connection_add_rate'] = 0.08
            neat_config['node_add_rate'] = 0.03

    def _inject_random_individuals(self, population, config, percent=0.1):
        """Replaces the worst individuals with new, random ones."""
        # Calculate average fitness
        fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
        avg_fitness = np.mean(fitnesses) if fitnesses else 0
        
        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf'))
        
        num_to_inject = int(len(population) * percent)
        injected = 0
        
        for i in range(len(population)):
            if injected >= num_to_inject:
                break
            # Only replace individuals below average
            if population[i].fitness is None or population[i].fitness < avg_fitness:
                body, connections = sample_robot(config['robot_size'])
                new_individual = NEATIndividual(body, connections, neat_config=config)
                new_individual.fitness = population[i].fitness if population[i].fitness is not None else -1000
                population[i] = new_individual
                injected += 1

    def _dynamic_compatibility_adjustment(self, species_manager, neat_config):
        """Dynamically adjust threshold every generation"""
        if not species_manager or len(species_manager.species) == 0:
            return
            
        current_count = len(species_manager.species)
        target = neat_config.get('target_species', 6)
        
        # EMERGENCY: If stuck at 1 species for too long, force speciation
        if current_count == 1 and len(self.best_fitness_history) > 5:
            neat_config['compatibility_threshold'] *= 0.5  # Dramatic reduction
            self.logger.warning(f"EMERGENCY: Forcing speciation! Threshold → {neat_config['compatibility_threshold']:.2f}")
        elif current_count < target - 2:
            neat_config['compatibility_threshold'] *= 0.90
            self.logger.info(f"Reducing compatibility threshold to {neat_config['compatibility_threshold']:.2f}")
        else:
            # Gentle adjustment when close
            if current_count < target:
                neat_config['compatibility_threshold'] *= 0.98
            elif current_count > target:
                neat_config['compatibility_threshold'] *= 1.02
        
        # Bounds
        neat_config['compatibility_threshold'] = np.clip(
            neat_config['compatibility_threshold'], 2.0, 10.0  # Lower minimum
        )

    def _apply_pressure_waves(self, generation, neat_config):
        """Apply oscillating parameter changes to prevent local optima"""
        freq = self.scheduler_config.get('pressure_wave_frequency')
        if not freq or freq <= 0:
            return
        
        # Create sine wave oscillation
        wave_phase = (generation % freq) / freq
        oscillation = np.sin(2 * np.pi * wave_phase)
        
        # Apply oscillation to mutation power
        base_mutation_power = self.baseline_config['neat_config']['weight_mutation_power']
        neat_config['weight_mutation_power'] = base_mutation_power * (1 + 0.3 * oscillation)