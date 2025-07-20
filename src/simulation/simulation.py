# src/simulation/simulation.py
"""
Simulation class for running EvolutionGym experiments.
"""

import numpy as np
from datetime import datetime
import imageio
from evogym import sample_robot
import pickle
from pathlib import Path        
import multiprocessing
from functools import partial

from src.simulation.individual import RandomIndividual, NEATIndividual
from .evolution import create_next_generation
from .evaluation import evaluate_individual_worker, evaluate_phenotype
from src.neat.species import SpeciesManager
from src.simulation.reporting import Reporter

class Simulation:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.population = []
        self.generation = 0
        self.species_manager = None
        self.checkpoint_path = Path("output/checkpoint.pkl")
        self.reporter = Reporter(output_dir=Path("output"))

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
        
        # Prepare the evaluation function with config
        eval_func = partial(
            evaluate_individual_worker,
            env_name=self.config['env'],
            episode_steps=self.config['episode_steps'],
            fps=self.config['video_fps'],
            render_mode='none',
            log_level=self.config.get('log_level', 'INFO')
        )
        
        # Evaluate in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # NOTE: To stop, just kill the terminal - checkpoint will resume
            # I can't figure out how to gracefully stop multiprocessing workers
            # Nor can I stop the gym version spam coming from every worker
            updated_population = pool.map(eval_func, self.population)

        self.population = updated_population
        
        # Find best individual
        for i, individual in enumerate(self.population):
            if individual.fitness is not None and individual.fitness > best_fitness:
                best_fitness = individual.fitness
                best_individual = individual
                best_idx = i + 1

        return best_fitness, best_individual, best_idx
    
    def save_best_individual(self, best_individual, best_fitness):
        """Save best individual video and structure"""
        if self.config['save_videos'] and self.config['render']:
            _, frames = evaluate_phenotype(
                best_individual.body,
                best_individual.connections,
                controller=best_individual.controller,
                render_mode='video',
                env_name=self.config['env'],
                episode_steps=self.config['episode_steps'],
                fps=self.config['video_fps']
            )
            
            if frames:
                timestamp = datetime.now().strftime("%H%M%S")
                video_path = f"output/videos/f{best_fitness:+07.2f}_g{self.generation:02d}_{timestamp}.mp4"
                imageio.mimsave(video_path, frames, fps=self.config['video_fps'], macro_block_size=1)
                self.logger.info(f"Saved video to {video_path}")
        
        if self.generation % self.config['save_best_every'] == 0 or \
           self.generation == self.config['max_generations'] - 1:
            save_path = f"output/robots/best_g{self.generation:02d}_f{best_fitness:+07.2f}.npz"
            np.savez(save_path, body=best_individual.body, connections=best_individual.connections)
            self.logger.info(f"Saved best robot to {save_path}")

    def run(self):
        """Run the full simulation"""
        self.logger.info("--- Simulation run started ---")
        self.logger.info("Note: To stop simulation, kill terminal. Progress is saved via checkpoints.")
        
        start_gen = self.generation

        if start_gen == 0:
            self.logger.info("Initializing new population...")
            self.initialize_population()

        for gen in range(start_gen, self.config['max_generations']):
            self.generation = gen
            self.logger.info(f"\n=== Generation {gen + 1}/{self.config['max_generations']} ===")
            
            if gen > start_gen:
                self.population = create_next_generation(
                    self.population, self.species_manager, self.config
                )
            
            # Evaluate
            best_fitness, best_individual, best_idx = self.evaluate_population()
            
            fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
            avg_fitness = np.mean(fitnesses) if fitnesses else 0
            self.logger.info(
                f"Generation {gen} Stats - Best: {best_fitness:.2f} (Robot {best_idx}), "
                f"Avg: {avg_fitness:.2f}"
            )

            self.reporter.log_generation(gen, self.population, self.species_manager)
            
            self.logger.info("Saving best individual...")
            self.save_best_individual(best_individual, best_fitness)
            
            self.logger.info("Saving checkpoint...")
            self.save_checkpoint()
            self.logger.info(f"Finished Generation {gen}")

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