import sys
import unittest
from unittest.mock import patch, MagicMock
from sklearn.base import BaseEstimator
import numpy as np

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

from BrestCancer.mdoels.prediction import LoadModel, ModelPredictor, ILoadModel
from Test import test_info, test_warning, test_error, test_debug, test_critical

class MockModel(BaseEstimator):
    """
    Mock model to simulate predictions.
    """
    def predict(self, X):
        return np.array([1] * len(X))  # Dummy prediction, returning 1 for all instances

class TestLoadModel(unittest.TestCase):
    """
    Unit tests for the LoadModel class.
    """

    @patch("BrestCancer.mdoels.prediction.joblib.load")
    def test_successful_model_load(self, mock_joblib_load):
        """
        Test successful model loading.
        """
        test_debug("Starting test: test_successful_model_load")
        try:
            mock_model = MockModel()
            mock_joblib_load.return_value = mock_model

            loader = LoadModel()
            model = loader.call("path_to_model.pkl")

            self.assertIsNotNone(model, "Model should be loaded successfully.")
            self.assertIsInstance(model, BaseEstimator, "Loaded model should be a BaseEstimator instance.")
            test_info("Model loaded successfully and is an instance of BaseEstimator.")
        except Exception as e:
            test_error(f"Error in test_successful_model_load: {e}")
            raise
        test_debug("Finished test: test_successful_model_load")

    @patch("BrestCancer.mdoels.prediction.joblib.load", side_effect=FileNotFoundError)
    def test_model_file_not_found(self, mock_joblib_load):
        """
        Test model loading when file is not found.
        """
        test_debug("Starting test: test_model_file_not_found")
        try:
            loader = LoadModel()
            model = loader.call("path_to_nonexistent_model.pkl")

            self.assertIsNone(model, "Model should be None when the file is not found.")
            test_info("Model correctly not found as expected.")
        except Exception as e:
            test_error(f"Error in test_model_file_not_found: {e}")
            raise
        test_debug("Finished test: test_model_file_not_found")

    @patch("BrestCancer.mdoels.prediction.joblib.load", side_effect=Exception("Load error"))
    def test_model_load_exception(self, mock_joblib_load):
        """
        Test model loading when an exception occurs.
        """
        test_debug("Starting test: test_model_load_exception")
        try:
            loader = LoadModel()
            model = loader.call("path_to_model.pkl")

            self.assertIsNone(model, "Model should be None when an exception occurs.")
            test_info("Model load failed as expected due to an exception.")
        except Exception as e:
            test_error(f"Error in test_model_load_exception: {e}")
            raise
        test_debug("Finished test: test_model_load_exception")

class TestModelPredictor(unittest.TestCase):
    """
    Unit tests for the ModelPredictor class.
    """

    def setUp(self):
        """
        Set up the mock model loader for testing.
        """
        test_debug("Setting up TestModelPredictor")
        self.mock_model = MockModel()
        self.mock_loader = MagicMock(spec=ILoadModel)
        self.predictor = ModelPredictor(self.mock_loader)

    def test_successful_prediction(self):
        """
        Test successful prediction.
        """
        test_debug("Starting test: test_successful_prediction")
        try:
            self.mock_loader.call.return_value = self.mock_model
            X_test = np.array([[0, 1], [1, 0]])

            predictions = self.predictor.call("path_to_model.pkl", X_test)

            self.assertTrue((predictions == np.array([1, 1])).all(), "Predictions should be correct.")
            test_info("Prediction was successful and returned the expected results.")
        except Exception as e:
            test_error(f"Error in test_successful_prediction: {e}")
            raise
        test_debug("Finished test: test_successful_prediction")

    def test_model_not_loaded(self):
        """
        Test prediction when model loading fails.
        """
        test_debug("Starting test: test_model_not_loaded")
        try:
            self.mock_loader.call.return_value = None
            X_test = np.array([[0, 1], [1, 0]])

            with self.assertRaises(ValueError, msg="Model should raise ValueError when loading fails."):
                self.predictor.call("path_to_model.pkl", X_test)
            test_info("Model loading failed as expected and raised ValueError.")
        except ValueError as e:
            test_warning(f"Expected ValueError: {e}")
        except Exception as e:
            test_error(f"Error in test_model_not_loaded: {e}")
            raise
        test_debug("Finished test: test_model_not_loaded")

    def test_prediction_failure(self):
        """
        Test when prediction fails due to an exception.
        """
        test_debug("Starting test: test_prediction_failure")
        try:
            self.mock_loader.call.return_value = self.mock_model
            self.mock_model.predict = MagicMock(side_effect=Exception("Prediction error"))
            X_test = np.array([[0, 1], [1, 0]])

            with self.assertRaises(Exception, msg="Prediction should raise an exception."):
                self.predictor.call("path_to_model.pkl", X_test)
            test_info("Prediction failed as expected and raised an exception.")
        except Exception as e:
            test_error(f"Error in test_prediction_failure: {e}")
            raise
        test_debug("Finished test: test_prediction_failure")

if __name__ == "__main__":
    unittest.main()
