import logging

# Log directory path for model logs
models_log_path = "/home/alrashidissa/Desktop/BreastCancer/Logging/models.log"

# Create a logger for models
models_logger = logging.getLogger("Models")
models_logger.setLevel(logging.DEBUG)  # Set the desired level

# Create file handler for models_logger
models_file_handler = logging.FileHandler(models_log_path)
models_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
models_logger.addHandler(models_file_handler)

# Optionally, add console output
models_console_handler = logging.StreamHandler()
models_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
models_logger.addHandler(models_console_handler)

# Logging functions for Models
def models_info(message):
    models_logger.info(message)

def models_warning(message):
    models_logger.warning(message)

def models_error(message):
    models_logger.error(message)

def models_debug(message):
    models_logger.debug(message)

def models_critical(message):
    models_logger.critical(message)
