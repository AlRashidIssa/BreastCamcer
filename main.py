import sys
import os
import logging
from src.BrestCancer.ingestdata.download_data import Download
from src.BrestCancer.ingestdata.unzip_data import Unzip
from src.BrestCancer.ingestdata.ingesting import Ingest
from src.BrestCancer.ingestdata.split import Split
from src.BrestCancer.preprocess.features_selction import Selction
from src.BrestCancer.preprocess.clear import Clean
from src.BrestCancer.preprocess.cat_encoder import Encoder
from src.BrestCancer.mdoels.algorithms import GradientBoostingModel
from src.BrestCancer.entity.config import CONFIG



try:
    # Download the dataset
    Download().call(url=CONFIG.url, output_path=CONFIG.download) # type: ignore

    # Unzip the dataset
    Unzip().call(zip_path=CONFIG.zip_path, extract_to=CONFIG.extract_to)

    # Ingest the dataset
    df = Ingest().call(path_csv=CONFIG.DFP)

    # Feature selection
    df = Selction().call(df=df, drop_columns=CONFIG.drop_columns)

    # Data cleaning
    df = Clean(fill_value=CONFIG.fill_value).call(
        df=df,
        drop_duplicates=CONFIG.drop_duplicates,
        outliers=CONFIG.drop_outliers,
        missing_columns=CONFIG.missing_columns,
        method=CONFIG.missing_columns # type: ignore
    )

    # Data encoding
    df = Encoder().call(
        df=df,
        columns=CONFIG.encoder_columns,
        method=CONFIG.scaler_method,
        replce=CONFIG.replce,
        value_replace=CONFIG.value_replce
    )

    # Split data
    X_train, X_test, y_train, y_test = Split().call(df=df, target=CONFIG.target)

    # Train model
    model = GradientBoostingModel().train(X=X_train, y=y_train, model_path_s=CONFIG.model_path_s)

except Exception as e:
    print(f"An error occurred: {e}")
    raise e