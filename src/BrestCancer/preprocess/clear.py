import pandas as pd
import sys
from abc import ABC, abstractmethod
from typing import List, Union

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BrestCancer import BrestCancer_critical, BrestCancer_error, BrestCancer_info, BrestCancer_warning

from BrestCancer.preprocess.miss_value import FillMissingValues

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

    def call(self, 
             df: pd.DataFrame,
             drop_duplicates: bool = True,
             outliers: bool = True,
             handling: bool = False,
             fill_na: bool = False,
             handl_missing: bool = False,
             missing_column: Union[str, None] = None, 
             method: str = "mean",
             fill_value: Union[str, int, float, None] = None) -> pd.DataFrame:
        """
        Cleans the DataFrame by performing various data preprocessing tasks, including removing duplicates,
        handling missing values, and detecting/removing outliers.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to be cleaned.

        drop_duplicates : bool, optional, default=True
            If True, duplicate rows in the DataFrame will be removed.

        outliers : bool, optional, default=True
            If True, outliers will be detected and removed using the Interquartile Range (IQR) method.

        handling : bool, optional, default=False
            Controls whether the handling of missing values should be done.

        fill_na : bool, optional, default=False
            If True, missing values will be filled in the DataFrame with zeroes.

        handl_missing : bool, optional, default=False
            If True, missing values will be handled (imputed) in specified columns. The method of imputation is determined by the `method` parameter.

        missing_column : str or None, optional, default=None
            A  column where missing values should be handled. If None, missing values in all columns will be processed.

        method : str, optional, default='mean'
            The method to use for filling missing values. Supported options include:
            - 'mean': Replace missing values with the mean of the column.
            - 'median': Replace missing values with the median of the column.
            - 'mode': Replace missing values with the mode (most frequent value) of the column.

        fill_value : str, int, float, or None, optional, default=None
            The value to fill missing values with. If None, the method parameter is used to determine the fill value.

        Returns:
        -------
        pd.DataFrame
            The cleaned DataFrame with duplicates removed, outliers handled, and missing values filled as specified.
        """
        try:
            # Step 1: Remove duplicates if specified
            if drop_duplicates:
                initial_row_count = df.shape[0]
                df.drop_duplicates(inplace=True)
                final_row_count = df.shape[0]
                BrestCancer_info(f"Removed {initial_row_count - final_row_count} duplicate rows")

            # Step 2: Handle missing values using the provided strategy
            if handl_missing:
                df = FillMissingValues().call(df, 
                                              handling=handling,
                                              column=missing_column, 
                                              method=method,
                                              fill_value=fill_value)
            
            if fill_na:
                df.fillna(0, inplace=True)
            
            # Step 3: Detect and remove outliers using the IQR method
            if outliers:
                initial_row_count = df.shape[0]
                
                for column in df.select_dtypes(include=['float64', 'int64']).columns:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                
                final_row_count = df.shape[0]
                BrestCancer_info(f"Removed {initial_row_count - final_row_count} outlier rows using IQR method")
            
            BrestCancer_info("Data cleaning process completed successfully.")
            return df

        except Exception as e:
            BrestCancer_error(f"An error occurred during data cleaning: {e}")
            raise
