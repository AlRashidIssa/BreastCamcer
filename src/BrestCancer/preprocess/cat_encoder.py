import warnings
warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="ignore")


import pandas as pd
import sys
from abc import ABC, abstractmethod
from typing import List, Union
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BrestCancer import BrestCancer_critical, BrestCancer_error, BrestCancer_info, BrestCancer_warning

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
        """
        pass

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
            # Check if the DataFrame is empty
            if df.empty:
                BrestCancer_warning("The DataFrame is empty.")
                return df

            # Check if all specified columns are in the DataFrame
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                BrestCancer_error(f"Columns not found in DataFrame: {missing_cols}")
                raise ValueError(f"Columns not found: {', '.join(missing_cols)}")

            if replce:
                # Replace values in specified columns
                for col in columns:
                    if col in df.columns:
                        df[col] = df[col].replace(value_replace)
                BrestCancer_info("Values replaced according to the provided dictionary.")
            else:
                # Apply encoding method
                if method == "label":
                    le = SklearnLabelEncoder()
                    for col in columns:
                        if col in df.columns:
                            df[col] = le.fit_transform(df[col])
                    BrestCancer_info("Label encoding applied.")
                elif method == "onehot":
                    ohe = SklearnOneHotEncoder(sparse=False, drop='first')
                    encoded = ohe.fit_transform(df[columns])
                    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(columns))
                    df = df.drop(columns=columns).join(encoded_df)
                    BrestCancer_info("One-hot encoding applied.")
                else:
                    BrestCancer_error(f"Unknown encoding method: {method}")
                    raise ValueError(f"Unknown method: {method}")

            return df
        
        except Exception as e:
            BrestCancer_error(f"An error occurred: {str(e)}")
            raise