import logging
import os

# Log directory path for preprocess logs
preprocess_log_path = "/home/alrashidissa/Desktop/BreastCancer/Logging/preprocess.log"

# Ensure the log directory exists
log_dir = os.path.dirname(preprocess_log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a logger for preprocessing
preprocess_logger = logging.getLogger("preprocess")
preprocess_logger.setLevel(logging.DEBUG)  # Set the desired level

# Create a file handler for preprocess_logger
try:
    preprocess_file_handler = logging.FileHandler(preprocess_log_path)
    preprocess_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    preprocess_logger.addHandler(preprocess_file_handler)
except Exception as e:
    print(f"Failed to create log file handler: {e}")

# Optionally, add console output
preprocess_console_handler = logging.StreamHandler()
preprocess_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
preprocess_logger.addHandler(preprocess_console_handler)

# Logging functions for preprocess
def preprocess_info(message):
    preprocess_logger.info(message)

def preprocess_warning(message):
    preprocess_logger.warning(message)

def preprocess_error(message):
    preprocess_logger.error(message)

def preprocess_debug(message):
    preprocess_logger.debug(message)

def preprocess_critical(message):
    preprocess_logger.critical(message)


