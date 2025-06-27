# evogym-playground/src/utils/logger.py

import logging
import os
from datetime import datetime

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
