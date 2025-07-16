# src/utils/bootstrap.py
"""
Project bootstrap
"""

import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    # Basic experiment settings
    "population_size": 100,
    "max_generations": 150,
    "env": "Walker-v0",
    "episode_steps": 500,
    "render": True,
    "log_level": "INFO",
    
    # Robot constraints
    "robot_size": [5, 5],
    
    # Evolution parameters
    "mutation_rate": 0.1,
    "mutation_amount": 0.3,
    "crossover_rate": 0.7,
    "elitism": 1,
    "tournament_size": 3,
    
    # Control parameters
    "control_type": "neat",  # "random", "neat", "hyperneat"

    "neat_config": {
        "compatibility_threshold": 3.0,
        "target_species": 10,
        "excess_coefficient": 1.0,
        "disjoint_coefficient": 1.0,
        "weight_coefficient": 0.5,
        "crossover_rate": 0.8,
        "stagnation_threshold": 20,
        
        "weight_mutation_rate": 0.8,
        "weight_perturb_rate": 0.9,
        "weight_mutation_power": 0.3,

        "connection_add_rate": 0.08,
        "node_add_rate": 0.04,        
    },    
    
    # Output settings
    "save_best_every": 5,
    "save_videos": True,
    "video_fps": 30
}

DEFAULT_DIRECTORIES = [
    'logs',
    'output/images',
    'output/videos',
    'output/plots',
    'output/robots',
]

def load_config(config_path='config.json'):
    """
    Load configuration from a JSON file, or create a default one if it doesn't exist.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Configuration file {config_path} not found. Creating default configuration.")
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    
    with open(config_path, 'r') as f:
        return json.load(f)
    
def ensure_directories_exist(directories=None):
    """
    Ensure that all required directories exist.
    
    Args:
        directories (list): List of directory paths to ensure exist.
        
    Returns:
        None
    """
    if directories is None:
        directories = DEFAULT_DIRECTORIES
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Directory already exists: {path}")