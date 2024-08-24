import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/alrashidissa/Desktop/BreastCancer/Logging/mlflow.log"),
        logging.StreamHandler()  
    ]
)

# Create a custom logger
logger = logging.getLogger(__name__)

# Create a Logging Funcation

def mlflow_info(massage):
    logger.info(msg=massage)

def mlflow_warning(massage):
    logger.warning(msg=massage)

def mlflow_error(massage):
    logger.error(msg=massage)

def mlflow_debug(massage):
    logger.debug(massage)

def mlflow_critical(massage):
    logger.critical(msg=massage)