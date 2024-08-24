import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src.BreastCancer.components.PreprocessAndPrediction import APIPredict


class TestAPIPredict(unittest.TestCase):

    @patch('src.BreastCancer.components.PreprocessAndPrediction.Selction')  # Mock the Selction class
    @patch('src.BreastCancer.components.PreprocessAndPrediction.Clean')     # Mock the Clean class
    @patch('src.BreastCancer.components.PreprocessAndPrediction.LoadModel') # Mock the LoadModel class
    @patch('src.BreastCancer.components.PreprocessAndPrediction.Predict')   # Mock the Predict class
    def test_api_predict(self, MockPredict, MockLoadModel, MockClean, MockSelction):
        # Mocking data and functions
        MockSelction().call.return_value = pd.DataFrame(np.random.rand(10, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        MockClean().call.return_value = pd.DataFrame(np.random.rand(10, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        MockLoadModel().call.return_value = MagicMock()
        MockPredict().call.return_value = np.array([0, 1])

        # Instantiate APIPredict
        api_predict = APIPredict()

        # Create a mock DataFrame for input X
        X = pd.DataFrame(np.random.rand(10, 5), columns=['id', 'feature1', 'feature2', 'feature3', 'feature4'])

        # Call the API predict method
        result = api_predict.call('mock_model_path', X)

        # Check if the result is either "Benign", "Malignant", or "Mixed"
        self.assertIn(result, ["Benign", "Malignant", "Mixed"])

    def test_invalid_model_path(self):
        with self.assertRaises(FileNotFoundError):
            api_predict = APIPredict()
            api_predict.call('invalid_path', pd.DataFrame(np.random.rand(10, 5), columns=['id', 'feature1', 'feature2', 'feature3', 'feature4']))

    def test_invalid_input_data(self):
        with self.assertRaises(ValueError):
            api_predict = APIPredict()
            api_predict.call('mock_model_path', 'invalid_data')

if __name__ == '__main__':
    unittest.main()