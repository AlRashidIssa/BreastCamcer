import sys

from typing import Dict, Any
from sklearn.base import BaseEstimator

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src.utils.logging import info, error

def get_model_parameters(model: BaseEstimator) -> Dict[Any, Any]:
    """
    Extract the parameters of a trained model.

    :param model: The trained machine learning model.
    :return: Dictionary of model parameters.
    """
    try:
        # Extract model parameters
        params = model.get_params()
        info("Model parameters extracted successfully.")
        return params
    except Exception as e:
        error(f"Error extracting model parameters: {e}")
        raise
    