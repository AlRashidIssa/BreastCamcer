import logging

# Log directory path for test logs
test_log_path = "/home/alrashidissa/Desktop/BreastCancer/Logging/test.log"

# Create a logger for tests
test_logger = logging.getLogger("Test")
test_logger.setLevel(logging.DEBUG)  # Set the desired level

# Create file handler for test_logger
test_file_handler = logging.FileHandler(test_log_path)
test_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
test_logger.addHandler(test_file_handler)

# Optionally, add console output
test_console_handler = logging.StreamHandler()
test_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
test_logger.addHandler(test_console_handler)

# Logging functions for Test
def test_info(message):
    test_logger.info(message)

def test_warning(message):
    test_logger.warning(message)

def test_error(message):
    test_logger.error(message)

def test_debug(message):
    test_logger.debug(message)

def test_critical(message):
    test_logger.critical(message)
