# evogym-playground/main.py
"""
Experimenting with EvolutionGym.
"""
import evogym.envs # Required to override MuJo environments
from evogym import sample_robot
import argparse
from datetime import datetime
import numpy as np
import random

from src.utils.bootstrap import load_config, ensure_directories_exist
from src.utils.logger import setup_logger
from src.simulation.evolution import create_next_generation
from src.simulation.evaluation import evaluate_robot

def parse_arguments():
    parser = argparse.ArgumentParser(description="EvolutionGym Experiment")
    parser.add_argument('--config', type=str, default='config.json', 
                        help="Path to configuration file")
    
    parser.add_argument('--pop-size', type=int, default=None,
                        help="Population size")
    parser.add_argument('--generations', type=int, default=None,
                        help="Number of generations")
    parser.add_argument('--render', action='store_true',
                        help="Enable rendering")
    parser.add_argument('--no-render', action='store_true',
                        help="Disable rendering")
    parser.add_argument('--no-video', action='store_true',
                        help="Disable video saving even if render is on")
    parser.add_argument('--env', default=None,
                        help="Environment name (e.g., Walker-v0, Climber-v0)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Only override if explicitly provided
    if args.pop_size is not None:
        config['population_size'] = args.pop_size
    if args.generations is not None:
        config['max_generations'] = args.generations
    if args.env is not None:
        config['env'] = args.env
    if args.seed is not None:
        config['seed'] = args.seed
    if args.render:
        config['render'] = True
    elif args.no_render:
        config['render'] = False

    if args.no_video:
        config['save_videos'] = False
    
    return config

def main():
    logger = setup_logger()
    logger.info("Starting EvolutionGym Experiment")

    config = parse_arguments()
    ensure_directories_exist()
    
    # Set random seed if provided
    if 'seed' in config:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        logger.info(f"Random seed set to {config['seed']}")
    
    logger.info(f"Configuration: pop_size={config['population_size']}, "
                f"generations={config['max_generations']}, env={config['env']}")
    
    # Initialize population tracking
    population = []
    fitnesses = []
    
    # Evolution loop
    for generation in range(config['max_generations']):
        logger.info(f"\n=== Generation {generation + 1}/{config['max_generations']} ===")
        
        # Create population (random for first generation, evolved after)
        if generation == 0:
            population = [sample_robot(config['robot_size']) for _ in range(config['population_size'])]
        else:
            population = create_next_generation(population, fitnesses, config)
        
        # Evaluate population
        fitnesses = []
        best_fitness = -float('inf')
        best_robot = None
        best_robot_idx = -1
        
        for i, (body, connections) in enumerate(population):
            # Only render first robot of first generation
            show_robot = config['render'] and i == 0 and generation == 0
            
            fitness = evaluate_robot(
                body, connections, 
                env_name=config['env'],
                render_mode='human' if show_robot else 'none',
                episode_steps=config['episode_steps'],
                fps=config['video_fps']
            )
            
            fitnesses.append(fitness)
            logger.debug(f"Robot {i+1}: Fitness = {fitness:.2f}")
            
            # Check if this is the best so far
            if fitness > best_fitness:
                best_fitness = fitness
                best_robot = (body, connections)
                best_robot_idx = i + 1
        
        # Log generation statistics
        avg_fitness = np.mean(fitnesses)
        logger.info(f"Generation {generation + 1} - Best: {best_fitness:.2f} (Robot {best_robot_idx}), Avg: {avg_fitness:.2f}")
        
        # Save video of generation's best
        if config['save_videos'] and config['render'] and best_robot:
            timestamp = datetime.now().strftime("%H%M%S")
            video_path = f"output/videos/f{best_fitness:+07.2f}_g{generation:02d}_{timestamp}.mp4"
            evaluate_robot(
                best_robot[0], best_robot[1], 
                env_name=config['env'],
                render_mode='video',
                video_path=video_path,
                episode_steps=config['episode_steps'],
                fps=config['video_fps'],
                logger=logger
            )
        
        # Save best robot structure periodically
        if generation % config['save_best_every'] == 0 or generation == config['max_generations'] - 1:
            save_path = f"output/robots/best_g{generation:02d}_f{best_fitness:+07.2f}.npz"
            np.savez(save_path, body=best_robot[0], connections=best_robot[1])
            logger.info(f"Saved best robot to {save_path}")
    
    # Show the final best robot
    if config['render'] and best_robot:
        logger.info("\nRendering final best robot...")
        evaluate_robot(
            best_robot[0], best_robot[1], 
            env_name=config['env'],
            render_mode='human',
            episode_steps=config['episode_steps']
        )


if __name__ == "__main__":
    main()