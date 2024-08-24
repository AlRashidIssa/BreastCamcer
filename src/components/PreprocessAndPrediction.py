from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src.BreastCancer.models.prediction import LoadModel, Predict
from BreastCancer.preprocess.features_selection import Selction
from BreastCancer.preprocess.clean import Clean
from src.BreastCancer.preprocess.scaler import Scaler
from src import BrestCancer_critical, BrestCancer_info

class IAPIPredict(ABC):
    """
    Abstract base class for making predictions using a pre-trained model.

    This class defines a common interface for all prediction implementations.
    """

    @abstractmethod
    def call(self, model_path: str, X: pd.DataFrame) -> Any:
        """
        Execute the prediction process.

        Args:
            model_path (str): Path to the pre-trained model file.
            X (pd.DataFrame): Input features for prediction.

        Returns:
            Any: Prediction results.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

class APIPredict(IAPIPredict):
    """
    Concrete implementation of IAPIPredict for making predictions.

    This class performs the following steps:
    1. Selection of relevant features.
    2. Data cleaning and preprocessing.
    3. Data scaling using Min-Max scaling.
    4. Loading the pre-trained model.
    5. Making predictions using the model.
    """

    def call(self, model_path: str, X: pd.DataFrame) -> Any:
        """
        Execute the prediction process.
    
        Args:
            model_path (str): Path to the pre-trained model file.
            X (pd.DataFrame): Input features for prediction.
    
        Returns:
            Any: Prediction results.
    
        Raises:
            FileNotFoundError: If the model file is not found.
            ValueError: If the input data is invalid.
            Exception: For any other unexpected errors.
        """
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input X must be a pandas DataFrame.")
    
            BrestCancer_info("Starting prediction process.")
    
            # Step 1: Feature selection
            BrestCancer_info("Selecting relevant features.")
            df = Selction().call(X, drop_columns=["id"])
    
            # Step 2: Data cleaning and preprocessing
            BrestCancer_info("Cleaning and preprocessing data.")
            df = Clean().call(df=df, 
                              drop_duplicates=False,
                              outliers=False,
                              handl_missing=False,
                              fill_na=True,
                              fill_value=0)
    
            # Step 3: Load the pre-trained model
            BrestCancer_info(f"Loading model from {model_path}.")
            model = LoadModel().call(mdoel_path=model_path)
            
            if model is None:
                raise FileNotFoundError(f"Model file not found at {model_path}.")
    
            # Step 4: Make predictions
            BrestCancer_info("Making predictions using the loaded model.")
            predictions = Predict().call(model=model, X=df.values)  # type: ignore
    
            if np.all(predictions == 0):
                predictions = "Benign"
            elif np.all(predictions == 1):
                predictions = "Malignant"
            else:
                predictions = "Mixed"
    
            BrestCancer_info("Prediction process completed successfully.")
            return predictions
    
        except FileNotFoundError as e:
            BrestCancer_critical(f"Model file not found: {e}")
            raise
        
        except ValueError as e:
            BrestCancer_critical(f"Invalid input data: {e}")
            raise
        
        except Exception as e:
            BrestCancer_critical(f"An unexpected error occurred: {e}")
            raise
        