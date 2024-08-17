from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator # type: ignore
from typing import Any
import numpy as np
from BrestCancer import models_info, models_warning, models_error, models_debug, models_critical

class IPredict(ABC):
    """
    Interface for making predictions using a machine learning model.

    Defines the contract for predicting using a trained machine learning model.
    """
    @abstractmethod
    def predict(self, model: BaseEstimator, X: Any) -> Any:
        """
        Make predictions with the provided model.

        :param model: A trained machine learning model.
        :param X: Data for prediction.
        :return: Predicted labels.
        """
        pass

class ModelPredictor(IPredict):
    def predict(self, model: BaseEstimator, X: Any) -> np.ndarray:
        """
        Make predictions with the provided model.

        :param model: A trained machine learning model.
        :param X: Data for prediction.
        :return: Predicted labels.
        """
        try:
            predictions = model.predict(X) # type: ignore
            models_info("Model made predictions.")
            return predictions
        except Exception as e:
            models_error(f"Error making predictions: {e}")
            raise
