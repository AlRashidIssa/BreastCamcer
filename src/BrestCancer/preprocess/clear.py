import pandas as pd
import sys
from abc import ABC, abstractmethod
from typing import List, Union

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

from src.BrestCancer.preprocess import preprocess_info, preprocess_warning, preprocess_debug, preprocess_error
from src.BrestCancer.preprocess.miss_value import FillMissingValues

class IClean(ABC):
    """
    Abstract base class for cleaning a DataFrame.
    
    This class defines the structure for any class that wishes to implement 
    data cleaning operations, such as removing duplicates, handling missing values, 
    and detecting outliers.
    """
    
    @abstractmethod
    def call(self, 
             df: pd.DataFrame, 
             drop_duplicates: bool = True, 
             outliers: bool = True, 
             handl_missing: bool = False,
             missing_columns: Union[List[str], None] = None, 
             method: str = "mean") -> pd.DataFrame:
        """
        Abstract method for cleaning a DataFrame.
        
        Subclasses must implement this method to specify how cleaning operations should be handled.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to clean.
        - drop_duplicates (bool): If True, duplicates will be removed. Default is True.
        - outliers (bool): If True, outliers will be detected and removed using the IQR method. Default is True.
        - missing_columns (list of str or None): List of columns to handle missing values. If None, all columns with missing values will be processed.
        - method (str): The method to use for filling missing values. Supported values are 'mean', 'median', and 'mode'. Default is 'mean'.
        
        Returns:
        - pd.DataFrame: The cleaned DataFrame.
        """
        pass


class Clean(IClean):
    """
    Concrete implementation of IClean that performs various data cleaning operations.
    
    This class is responsible for:
    - Removing duplicate rows
    - Handling missing values using the FillMissingValues strategy
    - Detecting and removing outliers using the IQR (Interquartile Range) method
    """
    
    def __init__(self, fill_value: Union[str, int, float, None] = None):
        """
        Initializes the Clean class with a fill strategy for handling missing values.
        
        Parameters:
        - fill_value (str, int, float, or None): The value to fill missing values with. If None, use the method specified in the `call` method.
        """
        self.fill_value = fill_value

    def call(self, 
             df: pd.DataFrame,
             drop_duplicates: bool = True,
             outliers: bool = True,
             handl_missing: bool = False,
             missing_columns: Union[List[str], None] = None, 
             method: str = "mean") -> pd.DataFrame:
        """
        Cleans the DataFrame by removing duplicates, handling missing values, and detecting/removing outliers.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to clean.
        - drop_duplicates (bool): If True, duplicates will be removed. Default is True.
        - outliers (bool): If True, outliers will be detected and removed using the IQR method. Default is True.
        - missing_columns (list of str or None): List of columns to handle missing values. If None, all columns with missing values will be processed.
        - method (str): The method to use for filling missing values. Supported values are 'mean', 'median', and 'mode'. Default is 'mean'.
        
        Returns:
        - pd.DataFrame: The cleaned DataFrame.
        """
        try:
            # Step 1: Remove duplicates if specified
            if drop_duplicates:
                initial_row_count = df.shape[0]
                df.drop_duplicates(inplace=True)
                final_row_count = df.shape[0]
                preprocess_info(f"Removed {initial_row_count - final_row_count} duplicate rows")

            # Step 2: Handle missing values using the provided strategy
            if handl_missing:
                df = FillMissingValues(fill_value=self.fill_value).call(df, columns=missing_columns, method=method) # type: ignore

            # Step 3: Detect and remove outliers using the IQR method
            if outliers:
                initial_row_count = df.shape[0]
                
                for column in df.select_dtypes(include=['float64', 'int64']).columns: # type: ignore
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                
                final_row_count = df.shape[0]
                preprocess_info(f"Removed {initial_row_count - final_row_count} outlier rows using IQR method")
            
            preprocess_info("Data cleaning process completed successfully.")
            return df

        except Exception as e:
            preprocess_error(f"An error occurred during data cleaning: {e}")
            raise
