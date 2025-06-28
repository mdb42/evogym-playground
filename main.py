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
from src.simulation.simulation import Simulation

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
    
    # Run simulation
    sim = Simulation(config, logger)
    sim.run()


if __name__ == "__main__":
    main()