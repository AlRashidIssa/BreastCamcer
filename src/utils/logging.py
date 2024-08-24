import logging

# Log directory path for BrestCancer logs
log_path = "/home/alrashidissa/Desktop/BreastCancer/Logging/BrestCancer.log"

# Create a logger for BrestCancer
logger = logging.getLogger("BrestCancer")
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

# Logging functions for BrestCancer
def info(message):
    logger.info(message)

def warning(message):
    logger.warning(message)

def error(message):
    logger.error(message)

def debug(message):
    logger.debug(message)

def critical(message):
    logger.critical(message)
