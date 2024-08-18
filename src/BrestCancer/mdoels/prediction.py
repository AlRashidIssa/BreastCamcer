from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Any, Optional
import numpy as np
import joblib  # Assuming joblib is used for loading the model

import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BrestCancer.mdoels import models_info, models_warning, models_error, models_debug, models_critical

class IPredict(ABC):
    """
    Interface for making predictions using a machine learning model.

    Defines the contract for predicting using a trained machine learning model.
    """
    @abstractmethod
    def call(self, model: BaseEstimator, X: Any) -> np.ndarray:
        """
        Make predictions with the provided model.

        :param model: A trained machine learning model.
        :param X: Data for prediction.
        :return: Predicted labels as a NumPy array.
        """
        pass

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
            models_debug(f"Attempting to load model from {path_model}.")
            model = joblib.load(path_model)
            models_info(f"Model loaded successfully from {path_model}.")
            return model
        except FileNotFoundError:
            models_error(f"Model file not found at {path_model}.")
        except Exception as e:
            models_error(f"An error occurred while loading the model: {e}")
        return None

class ModelPredictor(IPredict):
    """
    Class that handles model predictions.

    Combines model loading and prediction functionalities.
    """

    def __init__(self, model_loader: ILoadModel):
        """
        Initialize with a model loader.

        :param model_loader: An instance of a class that implements ILoadModel.
        """
        models_debug("Initializing ModelPredictor with a model loader.")
        self.model_loader = model_loader

    def call(self, path_model: str, X: Any) -> np.ndarray:
        """
        Make predictions with the loaded model.

        :param path_model: Path to the model file.
        :param X: Data for prediction.
        :return: Predicted labels as a NumPy array.
        :raises: Exception if the model could not be loaded or predictions fail.
        """
        try:
            models_info(f"Starting prediction process with model from {path_model}.")
            model = self.model_loader.call(path_model)
            if model is None:
                models_critical("Model could not be loaded; raising ValueError.")
                raise ValueError("Model could not be loaded.")

            models_debug("Model loaded successfully. Attempting to make predictions.")
            predictions = model.predict(X)  # type: ignore
            models_info("Model made predictions successfully.")
            models_debug(f"Predictions: {predictions}")
            return predictions
        except ValueError as ve:
            models_warning(f"Prediction failed due to model loading issues: {ve}")
            raise
        except Exception as e:
            models_error(f"Error during prediction: {e}")
            raise