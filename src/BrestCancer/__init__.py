import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/alrashidissa/Desktop/BreastCancer/Logging/models.log"),
        logging.StreamHandler()  
    ]
)

# Create a custom logger
logger = logging.getLogger(__name__)

# Create a Logging Funcation

def models_info(massage):
    logger.info(msg=massage)

def models_warning(massage):
    logger.warning(msg=massage)

def models_error(massage):
    logger.error(msg=massage)

def models_debug(massage):
    logger.debug(massage)

def models_critical(massage):
    logger.critical(msg=massage)