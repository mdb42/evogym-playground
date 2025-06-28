# src/simulation/simulation.py
import numpy as np
from datetime import datetime
from evogym import sample_robot

from .evolution import create_next_generation
from .evaluation import evaluate_robot

class Simulation:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.population = []
        self.fitnesses = []
        self.generation = 0
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = [
            sample_robot(self.config['robot_size']) 
            for _ in range(self.config['population_size'])
        ]
        self.logger.info(f"Initialized population of {len(self.population)} robots")
        
    def evaluate_population(self):
        """Evaluate all individuals in current population"""
        self.fitnesses = []
        best_fitness = -float('inf')
        best_robot = None
        best_robot_idx = -1
        
        for i, (body, connections) in enumerate(self.population):
            # Only render first robot of first generation
            show_robot = self.config['render'] and i == 0 and self.generation == 0
            
            fitness = evaluate_robot(
                body, connections,
                env_name=self.config['env'],
                render_mode='human' if show_robot else 'none',
                episode_steps=self.config['episode_steps'],
                fps=self.config['video_fps']
            )
            
            self.fitnesses.append(fitness)
            self.logger.debug(f"Robot {i+1}: Fitness = {fitness:.2f}")
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_robot = (body, connections)
                best_robot_idx = i + 1
                
        return best_fitness, best_robot, best_robot_idx
    
    def save_best_robot(self, best_robot, best_fitness):
        """Save best robot video and structure"""
        # Save video
        if self.config['save_videos'] and self.config['render']:
            timestamp = datetime.now().strftime("%H%M%S")
            video_path = f"output/videos/f{best_fitness:+07.2f}_g{self.generation:02d}_{timestamp}.mp4"
            evaluate_robot(
                best_robot[0], best_robot[1],
                env_name=self.config['env'],
                render_mode='video',
                video_path=video_path,
                episode_steps=self.config['episode_steps'],
                fps=self.config['video_fps'],
                logger=self.logger
            )
        
        # Save structure
        if self.generation % self.config['save_best_every'] == 0 or \
           self.generation == self.config['max_generations'] - 1:
            save_path = f"output/robots/best_g{self.generation:02d}_f{best_fitness:+07.2f}.npz"
            np.savez(save_path, body=best_robot[0], connections=best_robot[1])
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
                self.population = create_next_generation(
                    self.population, self.fitnesses, self.config
                )
            
            # Evaluate
            best_fitness, best_robot, best_idx = self.evaluate_population()
            
            # Log stats
            avg_fitness = np.mean(self.fitnesses)
            self.logger.info(
                f"Generation {gen} - Best: {best_fitness:.2f} (Robot {best_idx}), "
                f"Avg: {avg_fitness:.2f}"
            )
            
            # Save best
            self.save_best_robot(best_robot, best_fitness)
        
        # Final render
        if self.config['render'] and best_robot:
            self.logger.info("\nRendering final best robot...")
            evaluate_robot(
                best_robot[0], best_robot[1],
                env_name=self.config['env'],
                render_mode='human',
                episode_steps=self.config['episode_steps']
            )