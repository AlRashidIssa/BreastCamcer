from abc import ABC, abstractmethod
import sys
import pandas as pd
import os
from typing import Dict, Tuple, Optional, Any

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BreastCancer import BrestCancer_critical, BrestCancer_debug, BrestCancer_error, BrestCancer_info, BrestCancer_warning

class IIngest(ABC):
    """
    Abstract base class for data ingestion.

    This class defines the interface for ingestion classes, which should
    provide a concrete implementation of the `call` method to read
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

class Ingest(IIngest):
    """
    Concrete implementation of the data ingestion.

    This class implements the `call` method to read data from a CSV file
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
            BrestCancer_error(f"File Ingest Data, Invalid Path: {path_csv}")
            raise FileNotFoundError(f"File not found: {path_csv}")

        try:
            BrestCancer_debug(f"Starting Ingest Data From {path_csv}")
            # Read the CSV file into a DataFrame
            df = pd.read_csv(path_csv)
            BrestCancer_info(f"Successfully ingested data from {path_csv}")
            return df
        except pd.errors.EmptyDataError:
            BrestCancer_warning(f"The file {path_csv} is empty.")
            raise ValueError(f"The file {path_csv} is empty.")
        except pd.errors.ParserError:
            BrestCancer_error(f"Error parsing the file {path_csv}.")
            raise ValueError(f"Error parsing the file {path_csv}.")
        except Exception as e:
            BrestCancer_critical(f"An unexpected error occurred: {str(e)}")
            raise RuntimeError(f"An unexpected error occurred: {str(e)}")