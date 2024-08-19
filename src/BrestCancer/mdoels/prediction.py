from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Any, Optional
import numpy as np
import joblib  # Assuming joblib is used for loading the model

import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BrestCancer import BrestCancer_critical, BrestCancer_error,BrestCancer_debug, BrestCancer_info, BrestCancer_warning

class IPredict(ABC):
    """
    Interface for making predictions using a machine learning model.

    Defines the contract for predicting using a trained machine learning model.
    """
    @abstractmethod
    def call(self, model: BaseEstimator, X: np.ndarray) -> pd.Series:
        """
        Make predictions with the provided model.

        :param model: A trained machine learning model.
        :param X: Data for prediction.
        :return: Predicted labels as a NumPy array.
        """
        pass

class Predict(IPredict):
    """
    Class that handles model predictions.

    Combines model loading and prediction functionalities.
    """
    def call(self, model: BaseEstimator, X: np.ndarray) -> pd.Series:
        """
        Make predictions.

        :param X(np.ndarray): Data for prediction.
        :return(pd.Series): Predicted labels as a pandas Series
        :raises: Exception if the model could not be loaded or predictions fail.
        """
        try:
            BrestCancer_info(f"Starting prediction process with model.")
            if model is None:
                models_critical("Model could not be loaded; raising ValueError.")
                raise ValueError("Model could not be loaded.")

            BrestCancer_debug("Model loaded successfully. Attempting to make predictions.")
            predictions = model.predict(X)  # type: ignore
            BrestCancer_info("Model made predictions successfully.")
            BrestCancer_debug(f"Predictions: {predictions}")
            return predictions
        except ValueError as ve:
            BrestCancer_warning(f"Prediction failed due to model loading issues: {ve}")
            raise
        except Exception as e:
            BrestCancer_error(f"Error during prediction: {e}")
            raise

class ILoadModel(ABC):
    """
    Interface for loading a machine learning model.

    Defines the contract for loading a model from a specified path.
    """
    @abstractmethod
    def call(self, path_model: str) -> Optional[BaseEstimator]:
        """
        Load the machine learning model from the given path.

        :param path_model: Path to the model file.
        :return: The loaded model, or None if loading fails.
        """
        pass

class LoadModel(ILoadModel):
    """
    Concrete implementation for loading a machine learning model using joblib.
    """

    def call(self, path_model: str) -> Optional[BaseEstimator]:
        """
        Load the model from a .pkl file using joblib.

        :param path_model: Path to the model file.
        :return: Loaded model if successful, None otherwise.
        """
        try:
            BrestCancer_debug(f"Attempting to load model from {path_model}.")
            model = joblib.load(path_model)
            BrestCancer_info(f"Model loaded successfully from {path_model}.")
            return model
        except FileNotFoundError:
            BrestCancer_error(f"Model file not found at {path_model}.")
        except Exception as e:
            BrestCancer_error(f"An error occurred while loading the model: {e}")
        return None

