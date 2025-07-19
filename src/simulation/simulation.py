# src/simulation/simulation.py
"""
Simulation class for running EvolutionGym experiments.
"""

import numpy as np
from datetime import datetime
import imageio
from evogym import sample_robot

from src.simulation.individual import RandomIndividual, NEATIndividual
from .evolution import create_next_generation
from .evaluation import evaluate_individual
from src.neat.species import SpeciesManager
import pickle
from pathlib import Path

class Simulation:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.population = []
        self.generation = 0
        self.species_manager = None
        self.checkpoint_path = Path("output/checkpoint.pkl")

        # Try to load from a checkpoint, otherwise initialize
        if not self.load_checkpoint():
            if self.config.get('control_type') == 'neat':
                self.species_manager = SpeciesManager(self.config.get('neat_config', {}))
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        
        control_type = self.config.get('control_type', 'random')
        
        for _ in range(self.config['population_size']):
            body, connections = sample_robot(self.config['robot_size'])
            
            if control_type == 'random':
                individual = RandomIndividual(body, connections)
            elif control_type == 'neat':
                individual = NEATIndividual(body, connections, neat_config=self.config)
            else:
                raise ValueError(f"Unknown control type: {control_type}")
                
            self.population.append(individual)
        
    def evaluate_population(self):
        """Evaluate all individuals in current population"""
        best_fitness = -float('inf')
        best_individual = None
        best_idx = -1
        
        for i, individual in enumerate(self.population):
            # Render first individual robot of first generation
            render_robot = self.config['render'] and i == 0 and self.generation == 0
            
            fitness, _ = evaluate_individual(
                individual,
                env_name=self.config['env'],
                render_mode='human' if render_robot else 'none',
                episode_steps=self.config['episode_steps'],
                fps=self.config['video_fps']
            )
            
            self.logger.debug(f"Robot {i}: Fitness = {fitness:.2f}")
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
                best_idx = i + 1
                
        return best_fitness, best_individual, best_idx
    
    def save_best_individual(self, best_individual, best_fitness):
        """Save best individual video and structure"""
        if self.config['save_videos'] and self.config['render']:
            _, frames = evaluate_individual(
                best_individual,
                render_mode='video',
                env_name=self.config['env'],
                episode_steps=self.config['episode_steps'],
                fps=self.config['video_fps'],
                update_fitness=False
            )
            
            # Save video
            if frames:
                timestamp = datetime.now().strftime("%H%M%S")
                video_path = f"output/videos/f{best_fitness:+07.2f}_g{self.generation:02d}_{timestamp}.mp4"
                imageio.mimsave(video_path, frames, fps=self.config['video_fps'], macro_block_size=1)
                self.logger.info(f"Saved video to {video_path}")
        
        # Save structure
        if self.generation % self.config['save_best_every'] == 0 or \
           self.generation == self.config['max_generations'] - 1:
            save_path = f"output/robots/best_g{self.generation:02d}_f{best_fitness:+07.2f}.npz"
            np.savez(save_path, body=best_individual.body, connections=best_individual.connections)
            self.logger.info(f"Saved best robot to {save_path}")
    
    def run(self):
        """Run the full simulation"""
        self.logger.info("--- Simulation run started ---")
        start_gen = self.generation

        if start_gen == 0:
            self.logger.info("Initializing new population for Generation 1...")
            self.initialize_population()
            self.logger.info("Population initialized.")

        for gen in range(start_gen, self.config['max_generations']):
            self.generation = gen
            self.logger.info(f"\n=== Starting Generation {gen + 1}/{self.config['max_generations']} ===")
            
            # Create/evolve population
            if gen > start_gen:
                self.logger.info("Creating next generation...")
                self.population = create_next_generation(self.population, self.species_manager, self.config)
                self.logger.info("Next generation created.")
            
            # Evaluate
            self.logger.info(f"Evaluating population of {len(self.population)} individuals...")
            best_fitness, best_individual, best_idx = self.evaluate_population()
            self.logger.info("Population evaluation complete.")
            
            fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
            avg_fitness = np.mean(fitnesses) if fitnesses else 0
            self.logger.info(
                f"Generation {gen} Stats - Best: {best_fitness:.2f} (Robot {best_idx}), "
                f"Avg: {avg_fitness:.2f}"
            )
            
            self.logger.info("Saving best individual...")
            self.save_best_individual(best_individual, best_fitness)
            
            self.logger.info("Saving checkpoint...")
            self.save_checkpoint()
            self.logger.info(f"Finished Generation {gen + 1}")

        self.logger.info("Simulation loop finished successfully")
        self.remove_checkpoint()

    def save_checkpoint(self):
        state = {
            'generation': self.generation,
            'population': self.population,
            'species_manager': self.species_manager
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        self.logger.info(f"Checkpoint saved for generation {self.generation}")

    def load_checkpoint(self):
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            self.generation = state['generation'] + 1
            self.population = state['population']
            self.species_manager = state['species_manager']
            self.logger.info(f"Checkpoint loaded. Resuming from generation {self.generation}.")
            return True
        return False
        
    def remove_checkpoint(self):
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            self.logger.info("Simulation complete. Checkpoint removed.")
