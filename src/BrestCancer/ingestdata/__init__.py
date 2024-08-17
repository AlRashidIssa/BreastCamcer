import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/alrashidissa/Desktop/BreastCancer/Logging/ingestdata.log"),
        logging.StreamHandler()  
    ]
)

# Create a custom logger
logger = logging.getLogger(__name__)

# Create a Logging Funcation

def ingest_info(massage):
    logger.info(msg=massage)

def ingest_warning(massage):
    logger.warning(msg=massage)

def ingest_error(massage):
    logger.error(msg=massage)

def ingest_debug(massage):
    logger.debug(massage)

def ingest_critical(massage):
    logger.critical(msg=massage)