import logging

# Log directory path for BrestCancer logs
BrestCancer_log_path = "/home/alrashidissa/Desktop/BreastCancer/Logging/BrestCancer.log"

# Create a logger for BrestCancer
BrestCancer_logger = logging.getLogger("BrestCancer")
BrestCancer_logger.setLevel(logging.DEBUG)  # Set the desired level

# Check if handlers already exist to avoid duplication
if not BrestCancer_logger.hasHandlers():
    # Create file handler for BrestCancer_logger
    BrestCancer_file_handler = logging.FileHandler(BrestCancer_log_path)
    BrestCancer_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    BrestCancer_logger.addHandler(BrestCancer_file_handler)

    # Optionally, add console output
    BrestCancer_console_handler = logging.StreamHandler()
    BrestCancer_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    BrestCancer_logger.addHandler(BrestCancer_console_handler)

# Logging functions for BrestCancer
def BrestCancer_info(message):
    BrestCancer_logger.info(message)

def BrestCancer_warning(message):
    BrestCancer_logger.warning(message)

def BrestCancer_error(message):
    BrestCancer_logger.error(message)

def BrestCancer_debug(message):
    BrestCancer_logger.debug(message)

def BrestCancer_critical(message):
    BrestCancer_logger.critical(message)
