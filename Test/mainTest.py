import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Assume the main function and related classes have been imported as follows:
from preprocess.scaler import Scaler
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
from src.visualization.ExploreDataAnalysis import BreastCancerAnalyzer
import main
class TestMainFunction(unittest.TestCase):

    @patch('your_module.CONFIG')
    @patch('your_module.Download')
    @patch('your_module.Unzip')
    @patch('your_module.BreastCancerAnalyzer')
    @patch('your_module.Ingest')
    @patch('your_module.Selction')
    @patch('your_module.Clean')
    @patch('your_module.Encoder')
    @patch('your_module.Split')
    @patch('your_module.Scaler')
    @patch('your_module.GradientBoostingModel')
    @patch('your_module.Predict')
    @patch('your_module.MetricsEvaluator')
    def test_main_successful_execution(self, MockMetricsEvaluator, MockPredict, MockGradientBoostingModel, MockScaler, MockSplit, MockEncoder, MockClean, MockSelction, MockIngest, MockBreastCancerAnalyzer, MockUnzip, MockDownload, MockCONFIG):
        """
        Test the main function for successful execution with mocked components.
        """
        
        # Mock configuration
        mock_config = MagicMock()
        mock_config.url = "http://example.com/data.zip"
        mock_config.download = "/path/to/download"
        mock_config.zip_path = "/path/to/zip"
        mock_config.extract_to = "/path/to/extract"
        mock_config.DFP = "/path/to/data.csv"
        mock_config.drop_columns = ['col1', 'col2']
        mock_config.fill_value = 0
        mock_config.drop_duplicates = True
        mock_config.drop_outliers = False
        mock_config.missing_columns = ['col3']
        mock_config.missing_method = 'mean'
        mock_config.encoder_columns = ['col4']
        mock_config.scaler_method = 'minmax'
        mock_config.replce = True
        mock_config.value_replce = {'A': 1, 'B': 0}
        mock_config.target = 'target_column'
        mock_config.model_path_s = '/path/to/model'

        MockCONFIG.return_value = mock_config

        # Mock other classes and their methods
        MockDownload().call.return_value = None
        MockUnzip().call.return_value = None

        mock_df = pd.DataFrame({
            'radius_mean': [1.0, 2.0],
            'texture_mean': [3.0, 4.0],
            'target_column': [0, 1]
        })

        MockBreastCancerAnalyzer().load_data.return_value = None
        MockBreastCancerAnalyzer().inspect_data.return_value = None
        MockBreastCancerAnalyzer().analyze.return_value = None
        MockIngest().call.return_value = mock_df
        MockSelction().call.return_value = mock_df
        MockClean().call.return_value = mock_df
        MockEncoder().call.return_value = mock_df
        MockSplit().call.return_value = (mock_df.drop(columns=['target_column']), mock_df.drop(columns=['target_column']), [0, 1], [0, 1])
        MockScaler().call.return_value = mock_df.drop(columns=['target_column']).to_numpy()
        MockGradientBoostingModel().train.return_value = MagicMock()
        MockPredict().call.return_value = [0, 1]
        MockMetricsEvaluator().call.return_value = {'accuracy': 1.0}

        # Run the main function
        with patch('your_module.sys.exit') as mock_exit:
            main()

            # Ensure sys.exit was not called
            mock_exit.assert_not_called()

        # Assertions to ensure methods were called
        MockDownload().call.assert_called_once_with(url=mock_config.url, output_path=mock_config.download)
        MockUnzip().call.assert_called_once_with(zip_path=mock_config.zip_path, extract_to=mock_config.extract_to)
        MockIngest().call.assert_called_once_with(path_csv=mock_config.DFP)
        MockSelction().call.assert_called_once()
        MockClean().call.assert_called_once()
        MockEncoder().call.assert_called_once()
        MockSplit().call.assert_called_once()
        MockScaler().call.assert_called()
        MockGradientBoostingModel().train.assert_called_once()
        MockPredict().call.assert_called_once()
        MockMetricsEvaluator().call.assert_called_once()

    @patch('your_module.CONFIG')
    def test_main_config_failure(self, MockCONFIG):
        """
        Test the main function to handle configuration loading failure.
        """
        # Simulate a configuration loading error
        MockCONFIG.side_effect = Exception("Configuration loading failed")

        with self.assertRaises(SystemExit) as cm:
            main()

        self.assertEqual(cm.exception.code, 1)


