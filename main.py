import sys
import os
import logging

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

from src.BrestCancer.ingestdata.download_data import Download
from src.BrestCancer.ingestdata.unzip_data import Unzip
from src.BrestCancer.ingestdata.ingesting import Ingest
from src.BrestCancer.ingestdata.split import Split
from src.BrestCancer.preprocess.features_selction import Selction
from src.BrestCancer.preprocess.clear import Clean
from src.BrestCancer.preprocess.cat_encoder import Encoder
from src.BrestCancer.mdoels.algorithms import GradientBoostingModel
from src.BrestCancer.entity.config import CONFIG
from src.BrestCancer.components.evaluation_metrics import MetricsEvaluator
from src.BrestCancer.mdoels.prediction import Predict

from src import BrestCancer_info, BrestCancer_critical

def __main__() -> None:
    try:
        # Download the dataset
        BrestCancer_info("Starting dataset download.")
        Download().call(url=CONFIG.url, output_path=CONFIG.download)
        BrestCancer_info(f"Dataset downloaded to {CONFIG.download}.")
        
        # Unzip the dataset
        BrestCancer_info("Starting dataset extraction.")
        Unzip().call(zip_path=CONFIG.zip_path, extract_to=CONFIG.extract_to)
        BrestCancer_info(f"Dataset extracted to {CONFIG.extract_to}.")
        
        # Ingest the dataset
        BrestCancer_info("Ingesting dataset.")
        df = Ingest().call(path_csv=CONFIG.DFP)
        BrestCancer_info(f"Dataset ingested from {CONFIG.DFP}.")
        
        # Feature selection
        BrestCancer_info("Performing feature selection.")
        df = Selction().call(df=df, drop_columns=CONFIG.drop_columns)
        BrestCancer_info("Feature selection completed.")
        
        # Data cleaning
        BrestCancer_info("Starting data cleaning.")
        df = Clean(fill_value=CONFIG.fill_value).call(
            df=df,
            drop_duplicates=CONFIG.drop_duplicates,
            outliers=CONFIG.drop_outliers,
            missing_columns=CONFIG.missing_columns,
            method=CONFIG.missing_columns  # type: ignore
        )
        BrestCancer_info("Data cleaning completed.")
        
        # Data encoding
        BrestCancer_info("Encoding categorical features.")
        df = Encoder().call(
            df=df,
            columns=CONFIG.encoder_columns,
            method=CONFIG.scaler_method,
            replce=CONFIG.replce,
            value_replace=CONFIG.value_replce
        )
        BrestCancer_info("Categorical features encoded.")
        
        # Split data
        BrestCancer_info("Splitting dataset into training and testing sets.")
        X_train, X_test, y_train, y_test = Split().call(df=df, target=CONFIG.target)
        BrestCancer_info("Data split completed.")
        
        # Train model
        BrestCancer_info("Training model.")
        model = GradientBoostingModel().train(X=X_train, y=y_train, model_path_s=CONFIG.model_path_s)
        BrestCancer_info(f"Model trained and saved to {CONFIG.model_path_s}.")
        
        # Predict
        BrestCancer_info("Making predictions.")
        y_pred = Predict().call(model=model, X=X_test)
        metrics = MetricsEvaluator().call(y_true=y_test, y_pred=y_pred)
        BrestCancer_info("Predictions completed.")
        
        # Print the metrics
        BrestCancer_info("Metrics calculated:")
        for key, value in metrics.items():
            BrestCancer_info(f"{key}: {value}")

    except Exception as e:
        BrestCancer_critical(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    __main__()
