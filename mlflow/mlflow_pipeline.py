import argparse
import pandas as pd
import mlflow
import sys
import os

# Set the MLflow tracking URI to the correct address
mlflow.set_tracking_uri("http://127.0.0.1:8080") # type: ignore

# Set or create an experiment
mlflow.set_experiment("MLflow BreastCancer") # type: ignore

# Add project directory to sys.path for imports
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

from abc import ABC, abstractmethod
from src.BreastCancer.preprocess.scaler import Scaler
from src.BreastCancer.ingestdata.download_data import Download
from src.BreastCancer.ingestdata.unzip_data import Unzip
from src.BreastCancer.ingestdata.ingesting import Ingest
from src.BreastCancer.ingestdata.split import Split
from src.BreastCancer.preprocess.features_selection import Selction
from src.BreastCancer.preprocess.clean import Clean
from src.BreastCancer.preprocess.cat_encoder import Encoder
from src.BreastCancer.models.algorithms import ChooseModel
from src.BreastCancer.entity.config import CONFIG
from src.BreastCancer.components.evaluation_metrics import MetricsEvaluator
from src.BreastCancer.models.prediction import Predict
from src.visualization.ExploreDataAnalysis import BreastCancerAnalyzer

from src.BreastCancer import BrestCancer_critical, BrestCancer_debug, BrestCancer_error, BrestCancer_info, BrestCancer_warning


class IBreastCancerPipeline(ABC):
    """
    Abstract base class for the Breast Cancer pipeline.
    """
    @abstractmethod
    def run(self, yaml_path: str, analyzer: bool = False):
        """
        Abstract method for executing the pipeline stages.
        """
        pass


class BreastCancerPipeline(IBreastCancerPipeline):
    """
    Pipeline for processing breast cancer data, training a model, and evaluating predictions.
    """
    def run(self, yaml_path: str, analyzer: bool = False):
        """
        Initialize the pipeline with the given configuration.
        
        Args:
            yaml_path (str): Path to the YAML configuration file.
            analyzer (bool): Flag to perform Exploratory Data Analysis (EDA) if True.
        """
        self.config = CONFIG(yaml_path=yaml_path)  # Initialize config
        self.df = None
        BrestCancer_info(f"Configuration loaded from: {yaml_path}")

        try:
            with mlflow.start_run(): # type: ignore
                BrestCancer_info("Starting Breast Cancer Pipeline...")

                # Download data
                Download().call(url=self.config.url, output_path=self.config.download)

                # Unzip data
                Unzip().call(zip_path=self.config.zip_path, extract_to=self.config.extract_to)

                # Ingest data
                self.df = Ingest().call(path_csv=self.config.DFP)

                # Perform Exploratory Data Analysis (if analyzer is True)
                if analyzer:
                    BCA = BreastCancerAnalyzer(filepath=self.config.DFP, output_dir=self.config.plotes_pathes)
                    data_inspect = BCA.inspect_data()
                    mlflow.log_text(str(data_inspect), "data_inspection.txt") # type: ignore
                    for plot in os.listdir(self.config.plotes_pathes):
                        mlflow.log_artifact(os.path.join(self.config.plotes_pathes, plot)) # type: ignore

                # Feature selection
                self.df = Selction().call(df=self.df, drop_columns=self.config.drop_columns)

                # Data cleaning
                self.df = Clean().call(
                    df=self.df,
                    handling=self.config.handling,
                    drop_duplicates=self.config.drop_duplicates,
                    outliers=self.config.drop_outliers,
                    fill_na=self.config.fill_na,
                    handl_missing=self.config.handl_missing,
                    missing_column=self.config.missing_column,
                    method=self.config.missing_method,
                    fill_value=self.config.fill_value
                )

                # Encoding categorical features
                self.df = Encoder().call(
                    df=self.df,
                    columns=self.config.encoder_columns,
                    method=self.config.method_encoder,
                    replce=self.config.replce,
                    value_replace=self.config.value_replce
                )

                # Splitting data into training and test sets
                X_train, X_test, y_train, y_test = Split().call(target=self.config.target, df=self.df)

                # Model selection and training
                chosen_model = ChooseModel().call(name_model=self.config.name_model)
                model = chosen_model.train(X=X_train, y=y_train, model_path_s=self.config.model_path_s)
                
                # Log model to MLflow
                mlflow.sklearn.log_model( # type: ignore
                    sk_model=model,
                    artifact_path=self.config.model_path_s,
                    input_example=X_train,
                    registered_model_name=self.config.model_path_s,
                )

                # Model prediction
                y_pred = Predict().call(model=model, X=X_test)

                # Metrics evaluation
                y_true = pd.Series(y_test)
                y_pred = pd.Series(y_pred)
                metrics = MetricsEvaluator().call(y_true=y_true, y_pred=y_pred)
                mlflow.log_metrics(metrics) # type: ignore

                # Log the pipeline components
                pipeline_components = {
                    "Download": Download,
                    "Unzip": Unzip,
                    "Ingest": Ingest,
                    "Exploratory Data Analysis": BreastCancerAnalyzer,
                    "Feature Selection": Selction,
                    "Data Cleaning": Clean,
                    "Scaling": Scaler,
                    "Splitting": Split,
                    "Modeling": ChooseModel,
                    "Prediction": Predict,
                    "Evaluation Metrics": MetricsEvaluator
                }
                mlflow.log_dict(pipeline_components, "pipeline_components.json") # type: ignore

        except Exception as e:
            BrestCancer_critical(f"Pipeline encountered an error: {e}")
            raise


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the Breast Cancer pipeline.")
    # parser.add_argument('--yaml_path', type=str, required=True, help='Path to the YAML configuration file.')
    # parser.add_argument('--analyzer', action='store_true', help='Perform Exploratory Data Analysis (EDA).')

    # args = parser.parse_args()
    
    pipeline = BreastCancerPipeline()
    pipeline.run(yaml_path="/home/alrashidissa/Desktop/BreastCancer/ConfigrationYaml/config.yaml", 
                 analyzer=True)
