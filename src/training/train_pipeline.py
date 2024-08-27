import argparse
import pandas as pd
import sys
from abc import ABC, abstractmethod

sys.path.append("/home/alrashid/Desktop/BreastCancer")
from src.data.data_pipeline import DataPipeline
from src.models.models import ChooseModel
from src.entity.config import CONFIG
from src.evaluation.evaluate_model import MetricsEvaluator
from src.models.prediction import Predict
from src.visualization.ExploreDataAnalysis import BreastCancerAnalyzer

# Import logging functions
from src.utils.logging import info, critical

class ITrainPipeline(ABC):
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

class TrainPipeline(ITrainPipeline):
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
        info(f"Configuration loaded from: {yaml_path}")

        try:
            info("Starting Pipeline...")
            # Get the data from DataPipeline
            data_pipeline = DataPipeline(config=self.config)
            X_train, X_test, y_train, y_test, df_row = data_pipeline.call()
            
            # Perform Exploratory Data Analysis (if analyzer is True)
            if analyzer:
                analyzer = BreastCancerAnalyzer(filepath=self.config.DFP, output_dir=self.config.plotes_pathes)
                analyzer.load_data()
                data_inspected = analyzer.inspect_data()
                analyzer.analyze()
                info(f"EDA completed. Results: {data_inspected}")

            # Model selection and training
            choos_model = ChooseModel().call(name_model=self.config.name_model)
            model = choos_model.train(X=X_train, y=y_train, model_path_s=self.config.model_path_s)

            # Model prediction
            y_pred = Predict().call(model=model, X=X_test)

            # Metrics evaluation
            y_true = pd.Series(y_test)
            y_pred = pd.Series(y_pred)
            metrics = MetricsEvaluator().call(y_true=y_true, y_pred=y_pred)
            info(f"Metrics evaluated: {metrics}")

        except Exception as e:
            critical(f"Pipeline encountered an error: {e}")
            raise

def main(yaml_path=None, analyzer=False):
    # If the function is called with arguments, use them
    # Otherwise, parse the arguments from the command line
    if yaml_path is None:
        parser = argparse.ArgumentParser(description="Run the Breast Cancer pipeline.")
        parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
        parser.add_argument('--analyzer', action='store_true', help='Perform Exploratory Data Analysis (EDA).')

        args = parser.parse_args()
        yaml_path = args.config
        analyzer = args.analyzer

    pipeline = TrainPipeline()
    pipeline.run(yaml_path=yaml_path, analyzer=analyzer)

if __name__ == "__main__":
    main()
