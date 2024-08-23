import unittest
import sys
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

# Import the BreastCancerPipeline class
from MLflow.mlflow_pipeline import BreastCancerPipeline

class TestBreastCancerPipeline(unittest.TestCase):

    @patch('MLflow.mlflow_pipeline.mlflow.start_run')
    @patch('MLflow.mlflow_pipeline.mlflow.log_params')
    @patch('MLflow.mlflow_pipeline.mlflow.log_metrics')
    @patch('MLflow.mlflow_pipeline.mlflow.log_artifact')
    @patch('MLflow.mlflow_pipeline.mlflow.log_dict')
    @patch('MLflow.mlflow_pipeline.mlflow.set_tag')
    @patch('MLflow.mlflow_pipeline.mlflow.sklearn.log_model')
    @patch('MLflow.mlflow_pipeline.mlflow.data.from_pandas')
    @patch('MLflow.mlflow_pipeline.Download')
    @patch('MLflow.mlflow_pipeline.Unzip')
    @patch('MLflow.mlflow_pipeline.Ingest')
    @patch('MLflow.mlflow_pipeline.Selction')
    @patch('MLflow.mlflow_pipeline.Clean')
    @patch('MLflow.mlflow_pipeline.Encoder')
    @patch('MLflow.mlflow_pipeline.Split')
    @patch('MLflow.mlflow_pipeline.ChooseModel')
    @patch('MLflow.mlflow_pipeline.Predict')
    @patch('MLflow.mlflow_pipeline.MetricsEvaluator')
    @patch('MLflow.mlflow_pipeline.CONFIG')
    def test_pipeline_run(self, mock_config, mock_metrics_evaluator, mock_predict, mock_choose_model, mock_split,
                          mock_encoder, mock_clean, mock_selction, mock_ingest, mock_unzip, mock_download,
                          mock_from_pandas, mock_log_model, mock_set_tag, mock_log_dict, mock_log_artifact,
                          mock_log_metrics, mock_log_params, mock_start_run):
        
        # Mock CONFIG object
        mock_config.return_value.url = "fake_url"
        mock_config.return_value.download = "fake_download_path"
        mock_config.return_value.zip_path = "fake_zip_path"
        mock_config.return_value.extract_to = "fake_extract_path"
        mock_config.return_value.DFP = "fake_dfp"
        mock_config.return_value.drop_columns = ["column1", "column2"]
        mock_config.return_value.handling = "handling_strategy"
        mock_config.return_value.drop_duplicates = True
        mock_config.return_value.drop_outliers = False
        mock_config.return_value.fill_na = True
        mock_config.return_value.handl_missing = "some_strategy"
        mock_config.return_value.missing_column = "missing_column"
        mock_config.return_value.missing_method = "method"
        mock_config.return_value.fill_value = "value"
        mock_config.return_value.encoder_columns = ["col1", "col2"]
        mock_config.return_value.method_encoder = "method"
        mock_config.return_value.replce = True
        mock_config.return_value.value_replce = {"old_value": "new_value"}
        mock_config.return_value.target = "target"
        mock_config.return_value.name_model = "model_name"
        mock_config.return_value.model_path_s = "model_path"
        mock_config.return_value.plotes_pathes = "/fake/plots_path"
        mock_config.return_value.tage_description = "test_tag"
        mock_config.return_value.tage_name = "tag_name"
        mock_config.return_value.tage_value = "tag_value"

        # Mock method returns
        mock_download().call.return_value = None
        mock_unzip().call.return_value = None
        mock_ingest().call.return_value = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        mock_selction().call.return_value = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        mock_clean().call.return_value = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        mock_encoder().call.return_value = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        mock_split().call.return_value = (pd.DataFrame({"feature1": [1, 2]}), pd.DataFrame({"feature1": [1, 2]}), 
                                          pd.Series([0, 1]), pd.Series([0, 1]))
        mock_choose_model().call.return_value.train.return_value = "trained_model"
        mock_predict().call.return_value = pd.Series([0, 1])
        mock_metrics_evaluator().call.return_value = ({"accuracy": 0.99}, "/fake/plot_path")

        mock_from_pandas.return_value = "mock_dataset"

        # Initialize pipeline and run
        pipeline = BreastCancerPipeline()
        pipeline.run(yaml_path="fake_yaml_path", analyzer=False)

        # Assertions to check if the main components are called
        mock_download().call.assert_called_once()
        mock_unzip().call.assert_called_once()
        mock_ingest().call.assert_called_once()
        mock_selction().call.assert_called_once()
        mock_clean().call.assert_called_once()
        mock_encoder().call.assert_called_once()
        mock_split().call.assert_called_once()
        mock_choose_model().call.assert_called_once()
        mock_predict().call.assert_called_once()
        mock_metrics_evaluator().call.assert_called_once()

        # Assertions to check MLflow logging
        mock_start_run.assert_called_once()
        mock_log_params.assert_called_once()
        mock_log_metrics.assert_called_once()
        mock_log_artifact.assert_called()
        mock_log_model.assert_called_once()
        mock_set_tag.assert_called()
        mock_log_dict.assert_called_once()
        mock_from_pandas.assert_called_once_with(mock_ingest().call.return_value)

    @patch('MLflow.mlflow_pipeline.BrestCancer_critical')
    @patch('MLflow.mlflow_pipeline.Download')
    @patch('MLflow.mlflow_pipeline.mlflow.start_run')
    @patch('MLflow.mlflow_pipeline.CONFIG')
    def test_pipeline_error_handling(self, mock_config, mock_start_run, mock_download, mock_critical):
        # Simulate an exception being raised during download
        mock_download().call.side_effect = Exception("Download failed")

        pipeline = BreastCancerPipeline()

        with self.assertRaises(Exception):
            pipeline.run(yaml_path="fake_yaml_path", analyzer=False)

        # Check that the critical logging is called
        mock_critical.assert_called_once_with("Pipeline encountered an error: Download failed")


if __name__ == '__main__':
    unittest.main()
