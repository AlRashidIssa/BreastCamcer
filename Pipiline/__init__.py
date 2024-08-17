import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/alrashidissa/Desktop/BreastCancer/Logging/pipiline.log"),
        logging.StreamHandler()  
    ]
)

# Create a custom logger
logger = logging.getLogger(__name__)

# Create a Logging Funcation

def pipiline_info(massage):
    logger.info(msg=massage)

def pipiline_warning(massage):
    logger.warning(msg=massage)

def pipiline_error(massage):
    logger.error(msg=massage)

def pipiline_debug(massage):
    logger.debug(massage)

def pipiline_critical(massage):
    logger.critical(msg=massage)