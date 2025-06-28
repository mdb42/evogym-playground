# evogym-playground/src/simulation/simulation.py
"""
Simulation class for running EvolutionGym experiments.
"""

import numpy as np
from datetime import datetime
import imageio
from evogym import sample_robot

from .individual import Individual
from .evolution import create_next_generation
from .evaluation import evaluate_individual, evaluate_phenotype

class Simulation:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.population = []
        self.generation = 0
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for _ in range(self.config['population_size']):
            body, connections = sample_robot(self.config['robot_size'])
            self.population.append(Individual(body, connections))
        self.logger.info(f"Initialized population of {len(self.population)} robots")
        
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
        for gen in range(self.config['max_generations']):
            self.generation = gen
            self.logger.info(f"\n=== Generation {gen + 1}/{self.config['max_generations']} ===")
            
            # Create/evolve population
            if gen == 0:
                self.initialize_population()
            else:
                self.population = create_next_generation(self.population, self.config)
            
            # Evaluate
            best_fitness, best_individual, best_idx = self.evaluate_population()
            
            # Log stats
            fitnesses = [ind.fitness for ind in self.population]
            avg_fitness = np.mean(fitnesses)
            self.logger.info(
                f"Generation {gen} - Best: {best_fitness:.2f} (Robot {best_idx}), "
                f"Avg: {avg_fitness:.2f}"
            )
            
            # Save best
            self.save_best_individual(best_individual, best_fitness)
        
        # Final render
        if self.config['render'] and best_individual:
            self.logger.info("\nRendering final best robot...")
            evaluate_phenotype(
                best_individual.body, 
                best_individual.connections,
                controller=best_individual.controller,
                env_name=self.config['env'],
                render_mode='human',
                episode_steps=self.config['episode_steps'],
                fps=self.config['video_fps']
            )