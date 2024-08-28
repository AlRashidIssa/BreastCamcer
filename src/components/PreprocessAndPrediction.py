from abc import ABC, abstractmethod
import os
from typing import Any, Optional
import numpy as np
import pandas as pd
import sys

sys.path.append("/home/alrashid/Desktop/BreastCancer")
from src.utils.logging import info, critical
from src.models.prediction import Predict, LoadModel
from src.data.preprocess import Clean, Scale
from src.features.feature_selection import Selection
print("Hello")
# Path to the latest model version
models_paths = os.listdir("/home/alrashid/Desktop/BreastCancer/models/versioned")
model_path = os.path.join("/home/alrashid/Desktop/BreastCancer/models/versioned", models_paths[-1])

class IAPIPredict(ABC):
    """
    Abstract base class for making predictions using a pre-trained model.
    """

    @abstractmethod
    def call(self, X: pd.DataFrame) -> Any:
        """
        Execute the prediction process on the provided data.

        Args:
            X (pd.DataFrame): The input data for making predictions.

        Returns:
            Any: The result of the prediction process.

        Raises:
            NotImplementedError: If not implemented by the subclass.
        """
        pass

class APIPredict(IAPIPredict):
    """
    Concrete implementation of IAPIPredict for making predictions.
    """

    def call(self, X: pd.DataFrame) -> Optional[str]:
        """
        Execute the prediction process on the provided data.

        Args:
            X (pd.DataFrame): The input data for making predictions.

        Returns:
            Optional[str]: A string indicating the prediction result ("Benign" or "Malignant").

        Raises:
            ValueError: If the input data is not a pandas DataFrame.
            FileNotFoundError: If the model file is not found or loaded correctly.
            Exception: For any other unexpected errors.
        """
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input X must be a pandas DataFrame.")
    
            info("Starting prediction process.")    
            
            # Step 1: Feature selection
            info("Selecting relevant features.")
            df = Selection().call(df=X, drop_columns=["id"])
    
            # Step 2: Data cleaning and preprocessing
            info("Cleaning and preprocessing data.")
            df = Clean().call(df=df, 
                              drop_duplicates=False,
                              outliers=False,
                              handl_missing=False,
                              fill_na=True,
                              fill_value=0)
    
            # Step 3: Load the pre-trained model
            info(f"Loading model from {model_path}.")
            model = LoadModel().call(mdoel_path=model_path)
            
            if model is None:
                raise FileNotFoundError("Model not loaded correctly.")
    
            # Step 4: Make predictions
            info("Making predictions using the loaded model.")
            predictions = Predict().call(model=model, X=df)  # Use the model's predict method
    
            # Post-process predictions
            if np.all(predictions == 0):
                result = "Benign"
            elif np.all(predictions == 1):
                result = "Malignant"
            else:
                result = "Unknown"  # Handle case where predictions are not consistent

            info(f"Prediction result: {result}.")
            return result
    
        except FileNotFoundError as e:
            critical(f"Model file not found or failed to load: {e}")
            raise
        except ValueError as e:
            critical(f"Invalid input data: {e}")
            raise
        except Exception as e:
            critical(f"An unexpected error occurred: {e}")
            raise
