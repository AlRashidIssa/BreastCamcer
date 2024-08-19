"""
This code defines an interface and a concrete class for handling missing values in a 
pandas DataFrame. The IMissingValueHandler interface defines the contract for handling missing 
values, while FillMissingValues provides an implementation that allows filling missing values 
using various strategies such as mean, median, or mode. It includes robust error handling and 
logging for better debugging and monitoring.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, List

import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src.BrestCancer.preprocess import preprocess_debug, preprocess_info, preprocess_warning, preprocess_error


class IMissingValueHandler(ABC):
    """
    Abstract base class representing the interface for handling missing values in a DataFrame.
    
    This class defines the structure for any class that wishes to implement
    missing value handling operations, such as identifying and filling missing values.
    """

    @abstractmethod
    def call(self, 
             df: pd.DataFrame, 
             columns: Union[List[str], None] = None, 
             method: str = 'mean') -> pd.DataFrame:
        """
        Abstract method to handle missing values in the provided DataFrame.
        
        Subclasses must implement this method to specify how missing values should be handled.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to process.
        - columns (list of str or None): List of columns to handle. If None, all columns with missing values will be processed.
        - method (str): The method to use for filling missing values. Supported values are 'mean', 'median', and 'mode'.
        
        Returns:
        - pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


class FillMissingValues(IMissingValueHandler):
    """
    Concrete implementation of IMissingValueHandler that fills missing values in a DataFrame.
    
    This class provides functionality to fill missing values in specified columns using a given strategy.
    """

    def __init__(self, fill_value: Union[str, int, float, None] = None):
        """
        Initializes the FillMissingValues handler with an optional fill value.
        
        Parameters:
        - fill_value (str, int, float, or None): The value to fill missing values with. If None, use the method specified in the `call` method.
        """
        self.fill_value = fill_value

    def call(self, df: pd.DataFrame, columns: Union[List[str], None] = None, method: str = "mean") -> pd.DataFrame:
        """
        Handles missing values in the provided DataFrame by filling them with a specified strategy.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to process.
        - columns (list of str or None): List of columns to handle. If None, all columns with missing values will be processed.
        - method (str): The method to use for filling missing values. Supported values are 'mean', 'median', and 'mode'.

        Returns:
        - pd.DataFrame: The DataFrame with missing values filled.
        
        Raises:
        - ValueError: If an unsupported fill method is provided.
        - Exception: Catches other exceptions to log and raise errors during the process.
        """
        try:
            if columns is None:
                # Find all columns with missing values
                columns = df.columns[df.isna().any()].tolist()

            for column in columns:
                preprocess_info(f"Handling missing values for column: {column}")

                if self.fill_value is not None:
                    # Fill missing values with the specified value
                    df[column].fillna(self.fill_value, inplace=True)
                    preprocess_debug(f"Filled missing values in column '{column}' with '{self.fill_value}'")
                else:
                    # Fill missing values based on the method
                    if method == 'mean':
                        fill_value = df[column].mean()
                    elif method == 'median':
                        fill_value = df[column].median()
                    elif method == 'mode':
                        fill_value = df[column].mode()[0]
                    else:
                        preprocess_error(f"Unsupported fill method: {method}")
                        raise ValueError(f"Unsupported fill method: {method}")

                    df[column].fillna(fill_value, inplace=True)
                    preprocess_debug(f"Filled missing values in column '{column}' using '{method}' method with value '{fill_value}'")

            return df

        except ValueError as ve:
            preprocess_error(f"ValueError occurred: {ve}")
            raise
        except KeyError as ke:
            preprocess_error(f"KeyError occurred: {ke}")
            raise
        except TypeError as te:
            preprocess_error(f"TypeError occurred: {te}")
            raise
        except Exception as e:
            preprocess_error(f"An unexpected error occurred: {e}")
            raise