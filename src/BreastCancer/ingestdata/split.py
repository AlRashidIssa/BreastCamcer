import sys
import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Any
from sklearn.model_selection import train_test_split


sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BreastCancer import BrestCancer_critical, BrestCancer_debug, BrestCancer_error, BrestCancer_warning


class ISplit(ABC):
    """
    Abstract base class for data splitting.

    This class defines the interface for classes that perform data splitting. It should
    provide a concrete implementation of the `call` method to split the data into
    training and testing sets.

    Methods
    -------
    call(target: str, path_csv: str) -> List[Any]
        Splits the data from the provided CSV file into training and testing sets.
    """
    
    @abstractmethod
    def call(self, target: str, path_csv: str) -> List[Any]:
        """
        Abstract method to split data from a CSV file into training and testing sets.

        Parameters
        ----------
        target : str
            The name of the column to be used as the target variable.
        path_csv : str
            The file path to the CSV file containing the data.

        Returns
        -------
        List[Any]
            A list containing the following elements:
            - X_train: Training feature set (numpy array)
            - X_test: Testing feature set (numpy array)
            - y_train: Training target variable (pandas Series)
            - y_test: Testing target variable (pandas Series)
        
        Raises
        ------
        FileNotFoundError
            If the provided file path does not exist.
        ValueError
            If the target column is not found in the dataset or the dataset is empty.
        """
        pass

class Split(ISplit):
    """
    Concrete implementation of the data splitting.

    This class implements the `call` method to split data from a CSV file into training
    and testing sets based on the specified target variable. It includes error handling
    and logging for debugging purposes.

    Methods
    -------
    call(target: str, path_csv: str) -> List[Any]
        Splits the data from the provided CSV file into training and testing sets.
    """
    
    def call(self, target: str, df: pd.DataFrame) -> List[Any]:
        """
        Reads data from a CSV file, splits it into training and testing sets, and returns the
        feature and target variables.

        Parameters
        ----------
        target : str
            The name of the column to be used as the target variable.
        path_csv : str
            The file path to the CSV file containing the data.

        Returns
        -------
        List[Any]
            A list containing the following elements:
            - X_train: Training feature set (numpy array)
            - X_test: Testing feature set (numpy array)
            - y_train: Training target variable (pandas Series)
            - y_test: Testing target variable (pandas Series)
        
        Raises
        ------
        FileNotFoundError
            If the provided file path does not exist.
        ValueError
            If the target column is not found in the dataset or the dataset is empty.
        """
        try:
            if df.empty:
                BrestCancer_warning("The dataset is empty.")
                raise ValueError("The dataset is empty.")
            
            # Check if the target column exists in the DataFrame
            if target not in df.columns:
                error_message = f"Target column '{target}' not found in the dataset."
                BrestCancer_error(error_message)
                raise ValueError(error_message)
            
            # Split the data into features and target
            X = df.drop(columns=[target])
            y = df[target]
            
            # Check if the DataFrame is empty
            if X.empty or y.empty:
                error_message = "The dataset is empty."
                BrestCancer_warning(error_message)
                raise ValueError(error_message)
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            BrestCancer_debug(f"Data split into training and testing sets.")
            return [X_train, X_test, y_train, y_test]

        except pd.errors.EmptyDataError:
            error_message = "The Data Frame is empty."
            BrestCancer_warning(error_message)
            raise ValueError(error_message)
        except pd.errors.ParserError:
            error_message = f"Error parsing the Data Frame."
            BrestCancer_error(error_message)
            raise ValueError(error_message)
        except Exception as e:
            critical_message = f"An unexpected error occurred: {str(e)}"
            BrestCancer_critical(critical_message)
            raise RuntimeError(critical_message)
