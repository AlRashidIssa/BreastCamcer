import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/alrashidissa/Desktop/BreastCancer/Logging/api.log"),
        logging.StreamHandler()  
    ]
)

# Create a custom logger
logger = logging.getLogger(__name__)

# Create a Logging Funcation

def api_info(massage):
    logger.info(msg=massage)

def api_warning(massage):
    logger.warning(msg=massage)

def api_error(massage):
    logger.error(msg=massage)

def api_debug(massage):
    logger.debug(massage)

def api_critical(massage):
    logger.critical(msg=massage)