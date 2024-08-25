import os
import sys
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/")
from src.utils.logging import info, error,warning, critical, debug

# Abstract base class for data ingest
class IIngest(ABC):
    """
    Abstract base class for data ingestion.

    This class defines the interface for ingestion classes, which should
    provide a concrete implementation of the call method to read
    data from a CSV file.

    Methods
    -------
    call(path_csv: str) -> pd.DataFrame
        Reads data from the provided CSV file path and returns a DataFrame.
    """

    @abstractmethod
    def call(self, path_csv: str) -> pd.DataFrame:
        """
        Abstract method to read data from a CSV file.

        Parameters
        ----------
        path_csv : str
            The file path to the CSV file.

        Returns
        -------
        pd.DataFrame
            The data read from the CSV file.
        
        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        pass

# Concrete implementation of IIngest
class Ingest(IIngest):
    """
    Concrete implementation of the data ingestion.

    This class implements the call method to read data from a CSV file
    and return it as a Pandas DataFrame. It includes error handling and
    logging for debugging purposes.

    Methods
    -------
    call(path_csv: str) -> pd.DataFrame
        Reads data from the provided CSV file path and returns a DataFrame.
    """

    def call(self, path_csv: str) -> pd.DataFrame:
        """
        Reads data from a CSV file and returns it as a DataFrame.

        Parameters
        ----------
        path_csv : str
            The file path to the CSV file.

        Returns
        -------
        pd.DataFrame
            The data read from the CSV file.

        Raises
        ------
        FileNotFoundError
            If the provided file path does not exist.
        Exception
            If any other error occurs during file reading.
        """
        # Check if the file exists
        if not os.path.exists(path_csv):
            error(f"File Ingest Data, Invalid Path: {path_csv}")
            raise FileNotFoundError(f"File not found: {path_csv}")

        try:
            debug(f"Starting Ingest Data From {path_csv}")
            # Read the CSV file into a DataFrame
            df = pd.read_csv(path_csv)
            info(f"Successfully ingested data from {path_csv}")
            return df
        except pd.errors.EmptyDataError:
            warning(f"The file {path_csv} is empty.")
            raise ValueError(f"The file {path_csv} is empty.")
        except pd.errors.ParserError:
            error(f"Error parsing the file {path_csv}.")
            raise ValueError(f"Error parsing the file {path_csv}.")
        except Exception as e:
            critical(f"An unexpected error occurred: {str(e)}")
            raise RuntimeError(f"An unexpected error occurred: {str(e)}")

# Abstract base class for data encoding
class IEncoder(ABC):
    """
    Abstract base class for data encoding.
    
    Defines the interface for encoding data in a pandas DataFrame.
    """

    @abstractmethod
    def call(self,
             df: pd.DataFrame,
             columns: List[str],
             method: str,
             replce: bool = False,
             value_replace: dict = {"a": 1, "b": 0}
             ) -> pd.DataFrame:
        """
        Abstract method for encoding data.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be encoded.
            columns (List[str]): The list of columns to encode.
            method (str): The encoding method to use ('label' or 'onehot').
            replce (bool, optional): Whether to replace existing values or not. Defaults to False.
            value_replace (dict, optional): Dictionary for value replacement. Defaults to {"a": 1, "b": 0}.

        Returns:
            pd.DataFrame: The DataFrame with encoded columns.
        
        Raises:
            ValueError: If the encoding method is not recognized or if columns are not found in the DataFrame.
        """
        pass

# Concrete implementation of IEncoder
class Encoder(IEncoder):
    """
    Concrete implementation of the IEncoder class.

    Provides methods for encoding data in a pandas DataFrame using either label encoding or one-hot encoding.
    """

    def call(self, 
             df: pd.DataFrame, 
             columns: List[str], 
             method: str, 
             replce: bool = False, 
             value_replace: dict = {"a": 1, "b": 0}
             ) -> pd.DataFrame:
        """
        Encode specified columns in the DataFrame based on the chosen method.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be encoded.
            columns (List[str]): The list of columns to encode.
            method (str): The encoding method to use ('label' or 'onehot').
            replce (bool, optional): Whether to replace existing values or not. Defaults to False.
            value_replace (dict, optional): Dictionary for value replacement. Defaults to {"a": 1, "b": 0}.

        Returns:
            pd.DataFrame: The DataFrame with encoded columns.

        Raises:
            ValueError: If the encoding method is not recognized or if columns are not found in the DataFrame.
        """
        try:
            if df.empty:
                warning("The DataFrame is empty.")
                return df

            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                error(f"Columns not found in DataFrame: {missing_cols}")
                raise ValueError(f"Columns not found: {', '.join(missing_cols)}")

            if replce:
                for col in columns:
                    if col in df.columns:
                        df[col] = df[col].replace(value_replace)
                info("Values replaced according to the provided dictionary.")
            else:
                if method == "label":
                    le = SklearnLabelEncoder()
                    for col in columns:
                        if col in df.columns:
                            df[col] = le.fit_transform(df[col])
                    info("Label encoding applied.")
                elif method == "onehot":
                    ohe = SklearnOneHotEncoder(drop=None)
                    encoded = ohe.fit_transform(df[columns])
                    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(columns))
                    df = df.drop(columns=columns).join(encoded_df)
                    info("One-hot encoding applied.")
                else:
                    error(f"Unknown encoding method: {method}")
                    raise ValueError(f"Unknown method: {method}")

            return df
        
        except Exception as e:
            error(f"An error occurred: {str(e)}")
            raise

# Abstract base class for data cleaning
class IClean(ABC):
    """
    Abstract base class for cleaning a DataFrame.
    
    Defines the structure for any class that implements data cleaning operations,
    such as removing duplicates, handling missing values, and detecting outliers.
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
        - handl_missing (bool): If True, missing values will be handled (imputed) in specified columns. Default is False.
        - missing_columns (list of str or None): List of columns to handle missing values. If None, all columns with missing values will be processed.
        - method (str): The method to use for filling missing values. Supported values are 'mean', 'median', and 'mode'. Default is 'mean'.
        
        Returns:
        - pd.DataFrame: The cleaned DataFrame.
        """
        pass

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
             method: str = 'mean',
             fill_value: Union[str, int, float, None] = None) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the provided DataFrame.
        
        Subclasses must implement this method to specify how missing values should be handled.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to process.
        - columns (list of str or None): List of columns to handle. If None, all columns with missing values will be processed.
        - method (str): The method to use for filling missing values. Supported values are 'mean', 'median', and 'mode'.
        - fill_value (str, int, float, or None): The value to fill missing values with. If None, the method parameter is used to determine the fill value.
        
        Returns:
        - pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


class FillMissingValues(IMissingValueHandler):
    """
    Concrete implementation of IMissingValueHandler that fills missing values in a DataFrame.
    
    This class provides functionality to fill missing values in specified columns using a given strategy.
    """

    def call(self,
             df: pd.DataFrame, 
             handling: bool = False,
             column: Union[str, None] = None,
             method: str = "mean",
             fill_value: Union[str, int, float, None] = None) -> pd.DataFrame:
        """
        Handles missing values in the provided DataFrame by filling them with a specified strategy.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to process.
        - handling (bool): Flag to determine if missing values should be handled. If False, the DataFrame is returned as-is.
        - column (str or None): Column to handle. If None, all columns with missing values will be processed.
        - method (str): The method to use for filling missing values. Supported values are 'mean', 'median', and 'mode'.
        - fill_value (str, int, float, or None): The value to fill missing values with. If None, the method parameter is used to determine the fill value.

        Returns:
        - pd.DataFrame: The DataFrame with missing values filled.
        
        Raises:
        - ValueError: If an unsupported fill method is provided.
        - Exception: Catches other exceptions to log and raise errors during the process.
        """
        try:
            if handling:
                # If column is None, apply the filling operation to all columns with missing values
                if column is None:
                    columns_to_fill = df.columns[df.isnull().any()]
                else:
                    columns_to_fill = [column]

                info(f"Handling missing values for columns: {columns_to_fill}")

                # Fill missing values based on the method for each column
                for col in columns_to_fill:
                    if method == 'mean':
                        fill_value = df[col].mean()
                    elif method == 'median':
                        fill_value = df[col].median()
                    elif method == 'mode':
                        fill_value = df[col].mode()[0]
                    else:
                        error(f"Unsupported fill method: {method}")
                        raise ValueError(f"Unsupported fill method: {method}")
                    df[col].fillna(fill_value, inplace=True)
                    debug(f"Filled missing values in column '{col}' using '{method}' method with value '{fill_value}'")
                return df
            else:
                return df

        except ValueError as ve:
            error(f"ValueError occurred: {ve}")
            raise
        except KeyError as ke:
            error(f"KeyError occurred: {ke}")
            raise
        except TypeError as te:
            error(f"TypeError occurred: {te}")
            raise
        except Exception as e:
            error(f"An unexpected error occurred: {e}")
            raise

# Concrete implementation of IClean
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
             handl_missing: bool = False,
             fill_na: bool = False,
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

        handl_missing : bool, optional, default=False
            If True, missing values will be handled (imputed) in specified columns. The method of imputation is determined by the `method` parameter.

        fill_na : bool, optional, default=False
            If True, missing values will be filled in the DataFrame with the specified `fill_value`.

        missing_column : str or None, optional, default=None
            A column where missing values should be handled. If None, missing values in all columns will be processed.

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
            if drop_duplicates:
                initial_row_count = df.shape[0]
                df.drop_duplicates(inplace=True)
                final_row_count = df.shape[0]
                info(f"Removed {initial_row_count - final_row_count} duplicate rows")

            if handl_missing:
                df = FillMissingValues().call(df, 
                                              handling=handl_missing,
                                              column=missing_column, 
                                              method=method,
                                              fill_value=fill_value)
            
            if fill_na:
                df.fillna(fill_value, inplace=True)
            
            if outliers:
                initial_row_count = df.shape[0]
                
                for column in df.select_dtypes(include=['float64', 'int64']).columns:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
                
                final_row_count = df.shape[0]
                info(f"Removed {initial_row_count - final_row_count} outlier rows")

            return df
        
        except Exception as e:
            error(f"An error occurred: {str(e)}")
            raise

# Abstract base class for scaling data
class IScale(ABC):
    """
    Abstract base class for data scaling.
    
    Defines the interface for scaling data in a pandas DataFrame.
    """
    
    @abstractmethod
    def call(self, 
             df: pd.DataFrame, 
             columns: List[str],
             method: str) -> pd.DataFrame:
        """
        Abstract method for scaling data.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be scaled.
            columns (List[str]): The list of columns to scale.
            method (str): The scaling method to use ('standard' or 'minmax').
        
        Returns:
            pd.DataFrame: The DataFrame with scaled columns.
        
        Raises:
            ValueError: If the scaling method is not recognized or if columns are not found in the DataFrame.
        """
        pass

# Concrete implementation of IScale
class Scale(IScale):
    """
    Concrete implementation of IScale that performs data scaling using specified methods.
    
    This class provides methods for scaling data in a pandas DataFrame using standardization or min-max scaling.
    """

    def call(self, 
             df: pd.DataFrame, 
             columns: List[str], 
             method: str) -> pd.DataFrame:
        """
        Scale specified columns in the DataFrame based on the chosen method.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be scaled.
            columns (List[str]): The list of columns to scale.
            method (str): The scaling method to use ('standard' or 'minmax').

        Returns:
            pd.DataFrame: The DataFrame with scaled columns.

        Raises:
            ValueError: If the scaling method is not recognized or if columns are not found in the DataFrame.
        """
        try:
            if df.empty:
                warning("The DataFrame is empty.")
                return df

            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                error(f"Columns not found in DataFrame: {missing_cols}")
                raise ValueError(f"Columns not found: {', '.join(missing_cols)}")

            if method == "standard":
                scaler = SklearnStandardScaler()
                df[columns] = scaler.fit_transform(df[columns])
                info("Standard scaling applied.")
            elif method == "minmax":
                scaler = SklearnMinMaxScaler()
                df[columns] = scaler.fit_transform(df[columns])
                info("Min-max scaling applied.")
            else:
                error(f"Unknown scaling method: {method}")
                raise ValueError(f"Unknown method: {method}")

            return df
        
        except Exception as e:
            error(f"An error occurred: {str(e)}")
            raise

# Abstract base class for train-test splitting
class ISplit(ABC):
    """
    Abstract base class for splitting data into training and testing sets.
    
    Defines the interface for splitting data in a pandas DataFrame.
    """
    
    @abstractmethod
    def call(self, 
             df: pd.DataFrame, 
             target_column: str,
             test_size: float = 0.2, 
             random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Abstract method for splitting data into training and testing sets.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be split.
            target_column (str): The name of the target column.
            test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.2.
            random_state (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training features, testing features, training target, and testing target.

        Raises:
            ValueError: If the target column is not found in the DataFrame or if test_size is not in the range (0, 1).
        """
        pass

# Concrete implementation of ISplit
class Split(ISplit):
    """
    Concrete implementation of ISplit that performs data splitting into training and testing sets.
    
    This class uses the train_test_split function from scikit-learn to split the DataFrame into training and testing sets.
    """
    
    def call(self, 
             df: pd.DataFrame, 
             target_column: str,
             test_size: float = 0.2, 
             random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the DataFrame into training and testing sets.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be split.
            target_column (str): The name of the target column.
            test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.2.
            random_state (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training features, testing features, training target, and testing target.

        Raises:
            ValueError: If the target column is not found in the DataFrame or if test_size is not in the range (0, 1).
        """
        try:
            if target_column not in df.columns:
                error(f"Target column not found in DataFrame: {target_column}")
                raise ValueError(f"Target column not found: {target_column}")
            
            if not (0 < test_size < 1):
                error(f"Test size must be between 0 and 1: {test_size}")
                raise ValueError(f"Invalid test_size: {test_size}")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            info("Data split into training and testing sets.")
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            error(f"An error occurred: {str(e)}")
            raise    