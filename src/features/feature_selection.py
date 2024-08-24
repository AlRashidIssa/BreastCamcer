from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import sys


sys.path.append("/home/alrashidissa/Desktop/BreastCancer/")
from src.utils.logging import info, error, critical

class ISelection(ABC):
    """
    Abstract base class representing the selection operation interface.
    
    This class defines the structure for any class that wishes to implement
    data selection operations, including reading a dataset and dropping specific columns.
    """

    @abstractmethod
    def call(self, drop_columns: List[str]) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.
        
        This method should load the data from the provided file path, drop the specified columns,
        and return the modified DataFrame.

        Parameters:
        - drop_columns (list of str): The list of column names to drop from the DataFrame.

        Returns:
        - pd.DataFrame: The DataFrame with the specified columns dropped.
        """
        pass

class Selection(ISelection):
    """
    Concrete implementation of the ISelection abstract class.
    
    This class implements the `call` method which reads a CSV file into a DataFrame,
    drops specified columns, and handles exceptions that may occur during the process.
    """

    def call(self, df: pd.DataFrame, drop_columns: List[str]) -> pd.DataFrame:
        """
        Loads data from the specified file path, drops the given columns, and returns the modified DataFrame.
        
        Parameters:
        - drop_columns (list of str): The list of column names to drop from the DataFrame.

        Returns:
        - pd.DataFrame: The DataFrame with the specified columns dropped.
        """

        try:
            # Check if all specified columns exist in the DataFrame
            missing_columns = [col for col in drop_columns if col not in df.columns]
            if missing_columns:
                critical(f"Columns not found in the DataFrame: {missing_columns}")
                raise NameError(f"The following columns were not found in the DataFrame: {missing_columns}")
            
            # Drop the specified columns
            df = df.drop(columns=drop_columns)
            info(f"Columns {drop_columns} dropped successfully")
            
            return df
        
        except FileNotFoundError as fnf_error:
            error(f"File not found error: {fnf_error}")
            raise
        except NameError as name_error:
            error(f"Column name error: {name_error}")
            raise
        except pd.errors.EmptyDataError as empty_data_error:
            error(f"Empty data error: {empty_data_error}")
            raise
        except Exception as e:
            error(f"An unexpected error occurred: {e}")
            raise
