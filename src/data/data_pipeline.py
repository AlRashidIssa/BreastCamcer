import pandas as pd
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from src.utils.logging import info, error
from src.data.load_data import Download
from src.utils.unzip_data import Unzip
from src.data.preprocess import (Ingest,
                                 Clean,
                                 Encoder,
                                 Scale,
                                 Split)
from src.entity.config import CONFIG
from src.features.feature_selection import Selection


class IDataPipeline(ABC):
    """
    Abstract base class for data pipeline interfaces.

    Subclasses must implement the `call` method to define the data processing pipeline.
    """

    @abstractmethod
    def call(self) -> None:
        """
        Execute the data pipeline process.

        This method should be implemented by subclasses to define the specific 
        data processing steps and return the final output.
        """
        pass


class DataPipeline(IDataPipeline):
    """
    Concrete implementation of the data pipeline.

    This class orchestrates the data processing pipeline, including downloading,
    unzipping, ingesting, cleaning, encoding, scaling, and splitting the data.

    Attributes:
        config (CONFIG): Configuration containing parameters for each step.
        ingest (Ingest): Ingest class instance for loading data.
        selection (Selection): Selection class instance for feature selection.
        clean (Clean): Clean class instance for data cleaning.
        encoder (Encoder): Encoder class instance for encoding categorical features.
        scale (Scale): Scale class instance for feature scaling.
        split (Split): Split class instance for splitting data into training and testing sets.
    """

    def __init__(self, config: CONFIG) -> None:
        """
        Initialize the DataPipeline with the provided configuration.

        Args:
            config (CONFIG): Configuration instance containing parameters for each step of the pipeline.
        """
        self.config = config

        try:
            info("Starting download process")
            Download().call(url=self.config.url, output_path=self.config.download)
            info("Download completed successfully")
        except Exception as e:
            error(f"Error during download: {e}")
            raise

        try:
            info("Starting unzip process")
            Unzip().call(zip_path=self.config.zip_path, extract_to=self.config.extract_to)
            info("Unzip completed successfully")
        except Exception as e:
            error(f"Error during unzip: {e}")
            raise

        self.ingest = Ingest()
        self.selection = Selection()
        self.clean = Clean()
        self.encoder = Encoder()
        self.scale = Scale()
        self.split = Split()

    def call(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.DataFrame]:
        """
        Execute the data processing pipeline.

        Steps include ingesting data, selecting features, cleaning data, encoding features,
        scaling data, and splitting into training and testing sets.

        Returns:
            tuple: A tuple containing the training and testing sets for features and target values 
                   (X_train, X_test, y_train, y_test), and  raw  DataFrames.
        """
        try:
            info("Starting data ingestion")
            df_row = self.ingest.call(path_csv=self.config.DFP)
            info("Data ingestion completed successfully")
        except Exception as e:
            error(f"Error during data ingestion: {e}")
            raise
        
        try:
            info("Starting feature selection")
            df = self.selection.call(df=df_row, drop_columns=self.config.drop_columns)
            info("Feature selection completed successfully")
        except Exception as e:
            error(f"Error during feature selection: {e}")
            raise

        try:
            info("Starting data cleaning")
            df = self.clean.call(df=df, 
                                 drop_duplicates=self.config.drop_duplicates,
                                 outliers=self.config.drop_outliers, 
                                 handl_missing=self.config.handl_missing,
                                 fill_na=self.config.fill_na,
                                 missing_column=self.config.missing_column,
                                 method=self.config.missing_method,
                                 fill_value=self.config.fill_value)
            info("Data cleaning completed successfully")
        except Exception as e:
            error(f"Error during data cleaning: {e}")
            raise

        try:
            info("Starting encoding")
            df = self.encoder.call(df=df, 
                                   columns=self.config.encoder_columns,
                                   method=self.config.method_encoder,
                                   replce=self.config.replce,
                                   value_replace=self.config.value_replce)
            info("Encoding completed successfully")
        except Exception as e:
            error(f"Error during encoding: {e}")
            raise

        try:
            info("Starting scaling")
            df = self.scale.call(df=df, columns=self.config.scaler_columns,
                                method=self.config.scaler_method)
            df.to_csv(path_or_buf="/home/alrashidissa/Desktop/BreastCancer/data/processed/processed.csv")
            info("Scaling completed successfully")
        except Exception as e:
            error(f"Error during scaling: {e}")
            raise

        try:
            info("Starting data splitting")
            X_train, X_test, y_train, y_test = self.split.call(df=df, 
                                                               target_column=self.config.target,
                                                               test_size=self.config.test_size,
                                                               random_state=self.config.random_state)
            info("Data splitting completed successfully")
            return X_train, X_test, y_train, y_test, df_row
        except Exception as e:
            error(f"Error during data splitting: {e}")
            raise
