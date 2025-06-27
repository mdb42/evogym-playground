# evogym-playground/src/utils/bootstrap.py
"""
Project bootstrap
"""
import os
import json
import sys
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "population_size": 20,
    "max_generations": 50,
    "env": "Walker-v0",
    "render": True,
    "log_level": "INFO"
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