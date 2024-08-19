"""
This code defines an abstract base class ISelction and a concrete class Selction that implements the abstract method call. 
The purpose of this class is to load a dataset from a CSV file, remove specified columns, and return the modified DataFrame. 
The code also includes logging functionality to report critical errors, warnings, and other debugging information during the data 
processing steps.

The call method checks if the provided file path exists, loads the CSV file into a pandas DataFrame, checks if the specified 
columns exist in the DataFrame, and drops them if they do. The method handles several potential exceptions, including file not 
found, key errors, and other issues, while logging appropriate messages using a custom logging setup.
"""

from abc import ABC, abstractmethod
from typing import Any, List
import pandas as pd
import os

import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src.BrestCancer.preprocess import preprocess_critical, preprocess_debug, preprocess_info, preprocess_warning, preprocess_error

class ISelction(ABC):
    """
    Abstract base class representing the selection operation interface.
    
    This class defines the structure for any class that wishes to implement
    data selection operations, including reading a dataset and dropping specific columns.
    """

    @abstractmethod
    def call(self, path_data: str, drop_columns: List[str]) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.
        
        This method should load the data from the provided file path, drop the specified columns,
        and return the modified DataFrame.

        Parameters:
        - path_data (str): The file path to the dataset (CSV file).
        - drop_columns (list of str): The list of column names to drop from the DataFrame.

        Returns:
        - pd.DataFrame: The DataFrame with the specified columns dropped.

        Raises:
        - FileNotFoundError: If the file at `path_data` does not exist.
        - NameError: If any of the columns in `drop_columns` do not exist in the DataFrame.
        """
        pass

class Selction(ISelction):
    """
    Concrete implementation of the ISelction abstract class.
    
    This class implements the `call` method which reads a CSV file into a DataFrame,
    drops specified columns, and handles exceptions that may occur during the process.
    """

    def call(self, path_data: str, drop_columns: List[str]) -> pd.DataFrame:
        """
        Loads data from the specified file path, drops the given columns, and returns the modified DataFrame.
        
        Parameters:
        - path_data (str): The file path to the dataset (CSV file).
        - drop_columns (list of str): The list of column names to drop from the DataFrame.

        Returns:
        - pd.DataFrame: The DataFrame with the specified columns dropped.

        Raises:
        - FileNotFoundError: If the file at `path_data` does not exist.
        - NameError: If any of the columns in `drop_columns` do not exist in the DataFrame.
        """

        try:
            # Check if the file exists
            if not os.path.exists(path_data):
                preprocess_critical(f"No such file in directory or the directory has changed: {path_data}")
                raise FileNotFoundError(f"File not found: {path_data}")
            
            # Load the data into a DataFrame
            df = pd.read_csv(path_data)
            preprocess_info(f"Data loaded successfully from {path_data}")

            # Check if all specified columns exist in the DataFrame
            missing_columns = [col for col in drop_columns if col not in df.columns]
            if missing_columns:
                preprocess_critical(f"Columns not found in the DataFrame: {missing_columns}")
                raise NameError(f"The following columns were not found in the DataFrame: {missing_columns}")
            
            # Drop the specified columns
            df = df.drop(columns=drop_columns)
            preprocess_info(f"Columns {drop_columns} dropped successfully")
            
            return df
        
        except FileNotFoundError as fnf_error:
            preprocess_error(f"File not found error: {fnf_error}")
            raise
        except NameError as name_error:
            preprocess_error(f"Column name error: {name_error}")
            raise
        except pd.errors.EmptyDataError as empty_data_error:
            preprocess_error(f"Empty data error: {empty_data_error}")
            raise
        except Exception as e:
            preprocess_error(f"An unexpected error occurred: {e}")
            raise
