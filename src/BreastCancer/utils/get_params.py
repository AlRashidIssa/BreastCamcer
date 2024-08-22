import sys

from typing import Dict, Any
from sklearn.base import BaseEstimator

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src.BreastCancer import BrestCancer_info, BrestCancer_error

def get_model_parameters(model: BaseEstimator) -> Dict[Any, Any]:
    """
    Extract  the parameters of a trained model.

    :param model: The trained machine learning model.
    """
    try:
        # Extract model parameters
        params = model.get_params()
        BrestCancer_info("Model parameters extraction Saccessfully.")
        
    
    except Exception as e:
        BrestCancer_error(f"Error Extract model parameters: {e}")
        raise
    return params
    