import numpy as np
import argparse
import os
import sys
from abc import ABC, abstractmethod

import mlflow
import pandas as pd

sys.path.append("/home/alrashid/Desktop/BreastCancer")

from src.entity.config import CONFIG
from src.utils.logging import critical, info

from src.data.load_data import Download
from src.utils.unzip_data import Unzip
from src.features.feature_selection import Selection

from src.data.preprocess import Ingest, Clean, Encoder, Scale, Split

from src.data.data_pipeline import DataPipeline

from src.models.models import ChooseModel
from src.models.prediction import Predict

from src.visualization.ExploreDataAnalysis import BreastCancerAnalyzer
from src.evaluation.evaluate_model import MetricsEvaluator

from src.mlflow_.utlis.schema import save_schema_to_json
from src.utils.get_params import get_model_parameters

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("MLflow BreastCancer")

class IMLflowPipeline(ABC):
    """
    Abstract base class for the Breast Cancer pipeline.
    """
    @abstractmethod
    def run(self, yaml_path: str, analyzer: bool = False):
        """
        Abstract method for executing the pipeline stages.

        Args:
            yaml_path (str): Path to the YAML configuration file.
            analyzer (bool): Flag to perform Exploratory Data Analysis (EDA) if True.
        """
        pass

class MLflowPipeline(IMLflowPipeline):
    """
    Pipeline for processing breast cancer data, training a model, and evaluating predictions.
    """
    def run(self, yaml_path: str):
        """
        Execute the pipeline with the given configuration.

        Args:
            yaml_path (str): Path to the YAML configuration file.
            analyzer (bool): Flag to perform Exploratory Data Analysis (EDA) if True.
        """
        # Initialize configuration and logging
        self.config = CONFIG(yaml_path=yaml_path)
        info(f"Configuration loaded from: {yaml_path}")

        try:
            with mlflow.start_run():
                info("Starting Breast Cancer Pipeline...")

                # Run the data pipeline
                data_pipeline = DataPipeline(config=self.config)
                X_train, X_test, y_train, y_test, df_row = data_pipeline.call()

                dataset_df = mlflow.data.from_pandas(df_row)
                mlflow.log_input(dataset=dataset_df, context="Breast Cancer", tags={
                    "experiment_name": "MLflow BreastCancer",
                    "dataset_source": f"{self.config.url}",
                    "model_name": f"{self.config.name_model}",
                    "version": f"{self.config.data_version}"
                })
                info("Data ingested successfully.")

                # Exploratory Data Analysis (EDA)
                if True:
                    output_dir = self.config.plotes_pathes
                    analyzer = BreastCancerAnalyzer(filepath=self.config.DFP, output_dir=output_dir)
                    analyzer.load_data()
                    data_inspected = analyzer.inspect_data()
                    analyzer.analyze()

                    # Log EDA results
                    mlflow.log_text(str(data_inspected), "data_inspection.txt")
                    for plot in os.listdir(output_dir):
                        plot_path = os.path.join(output_dir, plot)
                        mlflow.log_artifact(plot_path, artifact_path="plots")

                # Save and log split data
                self._save_and_log_split_data(X_train, X_test, y_train, y_test)

                # Model selection and training
                chosen_model = ChooseModel().call(name_model=self.config.name_model)
                model = chosen_model.train(X=X_train, y=y_train, model_path_s=self.config.model_path_s)

                params = get_model_parameters(model=model)
                mlflow.log_params(params=params)

                # Log model and metadata
                self._log_model_and_metadata(model)

                # Save schema and log predictions
                self._save_schema_and_log_predictions(model, X_test, y_test)

                # Save and log plots
                for path in os.listdir(self.config.plotes_pathes):
                    mlflow.log_artifact(local_path=os.path.join(self.config.plotes_pathes, path), artifact_path="plots")

        except Exception as e:
            critical(f"Pipeline encountered an error: {e}")
            raise

    def _save_and_log_split_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        """
        Save and log split dataset artifacts.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Testing labels.
        """
        split_dir = '/home/alrashid/Desktop/BreastCancer/data/processed'
        os.makedirs(split_dir, exist_ok=True)

        X_train.to_csv(os.path.join(split_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(split_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(split_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(split_dir, 'y_test.csv'), index=False)

        for file_name in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
            mlflow.log_artifact(os.path.join(split_dir, file_name))

    def _log_model_and_metadata(self, model):
        """
        Log the trained model and its metadata to MLflow.

        Args:
            model: The trained model to log.
        """
        artifact_directory = os.path.basename(self.config.model_path_s)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_directory,
            registered_model_name=self.config.name_model,
        )
        mlflow.set_tag(key=f"{self.config.tage_description} {self.config.name_model}", value=1)
        mlflow.set_tag(self.config.tage_name, self.config.tage_value)

    def _save_schema_and_log_predictions(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Save schema and log predictions and metrics.

        Args:
            model: The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
        """
        # Define and save schema
        schema = {
            "input": {
                "X_train": {
                    "type": "DataFrame",
                    "columns": [
                        "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", 
                        "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", 
                        "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 
                        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", 
                        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", 
                        "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", 
                        "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", 
                        "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
                    ]
                }
            },
            "output": {
                "predictions": {
                    "type": "Array",
                    "elements": "Float"
                }
            }
        }
        save_schema_to_json(schema=schema)
        mlflow.log_artifact('model_schema.json')

        X_test_nd = X_test.to_numpy()
        y_pred = Predict().call(model=model, X=X_test_nd)
        metrics, plot_path = MetricsEvaluator().call(y_true=pd.Series(y_test), y_pred=pd.Series(y_pred))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        # Log the pipeline components
        pipeline_components = {
            "Download": Download,
            "Unzip": Unzip,
            "Ingest": Ingest,
            "Exploratory Data Analysis": BreastCancerAnalyzer,
            "Feature Selection": Selection,
            "Data Cleaning": Clean,
            "Categorical Encoder": Encoder,
            "Scaling": Scale,
            "Splitting": Split,
            "Modeling": ChooseModel,
            "Prediction": Predict,
            "Evaluation Metrics": MetricsEvaluator
        }
        mlflow.log_dict(pipeline_components, "pipeline_components.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Breast Cancer pipeline.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    pipeline = MLflowPipeline()
    pipeline.run(yaml_path=args.config)
