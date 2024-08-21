import sys
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Union
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BrestCancer import BrestCancer_debug, BrestCancer_critical, BrestCancer_error, BrestCancer_info, BrestCancer_warning

class IScaler(ABC):
    """
    Abstract base class for data scaling.

    Defines the interface for scaling data in a pandas DataFrame.
    """

    @abstractmethod
    def call(self,
             df: pd.DataFrame,
             columns: List[str],
             method: str,
             ) -> pd.DataFrame:
        """
        Abstract method for scaling data.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be scaled.
            columns (List[str]): The list of columns to scale.
            method (str): The scaling method to use ('standard' or 'minmax').

        Returns:
            pd.DataFrame: The DataFrame with scaled columns.
        """
        pass

class Scaler(IScaler):
    """
    Concrete implementation of the IScaler class.

    Provides methods for scaling data in a pandas DataFrame using either standard scaling or min-max scaling.
    """

    def call(self, 
             df: pd.DataFrame, 
             columns: List[str], 
             method: str
             ) -> pd.DataFrame:
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
            # Check if the DataFrame is empty
            if df.empty:
                BrestCancer_warning("The DataFrame is empty.")
                return df

            # Check if all specified columns are in the DataFrame
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                BrestCancer_error(f"Columns not found in DataFrame: {missing_cols}")
                raise ValueError(f"Columns not found: {', '.join(missing_cols)}")

            # Apply scaling method
            if method == "standard":
                scaler = SklearnStandardScaler()
                df[columns] = scaler.fit_transform(df[columns])
                BrestCancer_info("Standard scaling applied.")
            elif method == "minmax":
                scaler = SklearnMinMaxScaler()
                df[columns] = scaler.fit_transform(df[columns])
                BrestCancer_info("Min-max scaling applied.")
            else:
                BrestCancer_error(f"Unknown scaling method: {method}")
                raise ValueError(f"Unknown method: {method}")

            return df
        
        except Exception as e:
            BrestCancer_error(f"An error occurred: {str(e)}")
            raise


if __name__ == "__main__":
    import pandas as pd
    # Sample DataFrame
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })

    # Create Scaler instance
    scaler = Scaler()

    # Apply standard scaling
    scaled_data_standard = scaler.call(data, columns=['feature1', 'feature2'], method='standard')
    print("Standard Scaled Data:")
    print(scaled_data_standard)

    # Apply min-max scaling
    scaled_data_minmax = scaler.call(data, columns=['feature1', 'feature2'], method='minmax')
    print("Min-Max Scaled Data:")
    print(scaled_data_minmax)

