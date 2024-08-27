from abc import ABC, abstractmethod
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.client import MlflowClient
import numpy as np
import pandas as pd
import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer/")
from src.utils.logging import info, critical, debug, error, warning
from src.models.prediction import Predict
from src.data.preprocess import (Clean, Scale)
from src.features.feature_selection import Selection

client = MlflowClient()

# Set the model name dynamically or statically
model_name = "MyModel"  # Set this to your model's name

# Get the latest version of the model
latest_version = client.get_latest_versions(model_name, stages=["None"])[-1].version

# Transition the latest version to 'Production'
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

# Load the latest production version of the model
model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

class IAPIPredict(ABC):
    """
    Abstract base class for making predictions using a pre-trained model.
    """

    @abstractmethod
    def call(self, X: pd.DataFrame) -> Any:
        """
        Execute the prediction process.
        """
        pass

class APIPredict(IAPIPredict):
    """
    Concrete implementation of IAPIPredict for making predictions.
    """

    def call(self, model_path: str, X: pd.DataFrame) -> Any:
        """
        Execute the prediction process.
        """
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input X must be a pandas DataFrame.")
    
            info("Starting prediction process.")    
            
            # Step 1: Feature selection
            info("Selecting relevant features.")
            df = Selection().call(X, drop_columns=["id"])
    
            # Step 2: Data cleaning and preprocessing
            info("Cleaning and preprocessing data.")
            df = Clean().call(df=df, 
                              drop_duplicates=False,
                              outliers=False,
                              handle_missing=False,
                              fill_na=True,
                              fill_value=0)
    
            # Step 3: Load the pre-trained model
            info(f"Loading model from {f"models:/{model_name}/Production"}.")
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
            
            if model is None:
                raise FileNotFoundError(f"Model file not found at {model_path}.")
    
            # Step 4: Make predictions
            info("Making predictions using the loaded model.")
            predictions = Predict().call(model=model, X=df.values)  # type: ignore
    
            if np.all(predictions == 0):
                predictions = "Benign"
            elif np.all(predictions == 1):
                predictions = "Malignant"

            info("Prediction process completed successfully.")
            return predictions
    
        except FileNotFoundError as e:
            critical(f"Model file not found: {e}")
            raise
        except ValueError as e:
            critical(f"Invalid input data: {e}")
            raise
        except Exception as e:
            critical(f"An unexpected error occurred: {e}")
            raise