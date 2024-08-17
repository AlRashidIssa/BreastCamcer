import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/alrashidissa/Desktop/BreastCancer/Logging/test.log"),
        logging.StreamHandler()  
    ]
)

# Create a custom logger
logger = logging.getLogger(__name__)

# Create a Logging Funcation

def test_info(massage):
    logger.info(msg=massage)

def test_warning(massage):
    logger.warning(msg=massage)

def test_error(massage):
    logger.error(msg=massage)

def test_debug(massage):
    logger.debug(massage)

def test_critical(massage):
    logger.critical(msg=massage)