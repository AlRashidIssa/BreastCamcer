import logging
import os

def setup_logging(log_path: str):
    """
    Sets up logging for the Breast Cancer project.

    Args:
        log_path (str): Path to the log file.
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create a logger
    logger = logging.getLogger("System")
    logger.setLevel(logging.DEBUG)  # Set the desired level

    # Check if handlers already exist to avoid duplication
    if not logger.hasHandlers():
        # Create file handler for logger
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Optionally, add console output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    return logger

# Log directory path for BreastCancer logs
log_path = "/app/logs/system.log"

# Setup logging
logger = setup_logging(log_path)

# Logging functions for BreastCancer
def info(message: str):
    logger.info(message)

def warning(message: str):
    logger.warning(message)

def error(message: str):
    logger.error(message)

def debug(message: str):
    logger.debug(message)

def critical(message: str):
    logger.critical(message)
