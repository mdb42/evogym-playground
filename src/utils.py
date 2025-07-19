# src/utils.py
import json
from pathlib import Path
import logging
import os
from datetime import datetime

DEFAULT_CONFIG = {
    "population_size": 100,
    "max_generations": 250,
    "env": "Walker-v0",
    "episode_steps": 1000,
    "render": True,
    "log_level": "INFO",
    "robot_size": [
        5,
        5
    ],
    "mutation_rate": 0.04,
    "mutation_amount": 0.2,
    "crossover_rate": 0.7,
    "elitism": 2,
    "tournament_size": 3,
    "control_type": "neat",
    "neat_config": {
        "interspecies_crossover_chance": 0.01,
        "compatibility_threshold": 3.0,
        "target_species": 10,
        "elitism_per_species": 2,
        "excess_coefficient": 1.0,
        "disjoint_coefficient": 1.0,
        "weight_coefficient": 0.5,
        "crossover_rate": 0.8,
        "stagnation_threshold": 25,
        "weight_mutation_rate": 0.8,
        "weight_perturb_rate": 0.9,
        "weight_mutation_power": 0.15,
        "connection_add_rate": 0.2,
        "node_add_rate": 0.08
    },
    "save_best_every": 5,
    "save_videos": True,
    "video_fps": 30
}

DEFAULT_DIRECTORIES = [
    'logs',
    'output/videos',
    'output/robots',
]

def load_config(config_path='config.json'):
    """
    Load configuration from a JSON file.
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
    """
    if directories is None:
        directories = DEFAULT_DIRECTORIES
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


class ColorFormatter(logging.Formatter):
    """
    Custom log formatter that adds color coding to console log messages.
    
    Attributes:
        COLORS (dict): Mapping of log levels to ANSI color codes
        RESET (str): ANSI code to reset text color
    """
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """
        Format the log record with appropriate color coding.
        
        Args:
            record: Log record to format
            
        Returns:
            str: Color-coded formatted log message
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def setup_logger(log_dir="logs"):
    """
    Set up and configure the logging system.
    
    Args:
        log_dir (str): Directory to store log files (default: "logs")
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Define log format
    formatter = logging.Formatter(
        "[%(levelname)s] [%(asctime)s] [%(module)s:%(funcName)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add color-coded console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter(
        "[%(levelname)s] [%(asctime)s] [%(module)s:%(funcName)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console_handler)

    # Add file handler for persistent logging
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_logger(name=None):
    """
    Get the existing logger by name, or the root logger if None.
    
    Args:
        name (str, optional): Logger name, typically the module name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)