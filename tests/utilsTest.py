import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from sklearn.base import BaseEstimator

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

from src.BreastCancer.utils.get_params import get_model_parameters, BrestCancer_info, BrestCancer_error, BrestCancer_critical
from src.BreastCancer.utils.size import Size

class MockModel(BaseEstimator):
    def get_params(self):
        return {'param1': 1, 'param2': 2}

class TestGetModelParameters(unittest.TestCase):
    @patch('src.BreastCancer.utils.get_params.BrestCancer_info')
    @patch('src.BreastCancer.utils.get_params.BrestCancer_error')
    def test_get_model_parameters_success(self, mock_error, mock_info):
        model = MockModel()
        params = get_model_parameters(model)
        self.assertEqual(params, {'param1': 1, 'param2': 2})
        mock_info.assert_called_once_with("Model parameters extracted successfully.")

    @patch('src.BreastCancer.utils.get_params.BrestCancer_info')
    @patch('src.BreastCancer.utils.get_params.BrestCancer_error')
    def test_get_model_parameters_exception(self, mock_error, mock_info):
        model = MagicMock()
        model.get_params.side_effect = Exception("Test exception")
        with self.assertRaises(Exception):
            get_model_parameters(model)
        mock_error.assert_called_once_with("Error extracting model parameters: Test exception")

class TestSize(unittest.TestCase):
    @patch('src.BreastCancer.utils.size.BrestCancer_info')
    @patch('src.BreastCancer.utils.size.BrestCancer_error')
    @patch('src.BreastCancer.utils.size.BrestCancer_critical')
    def test_call_success(self, mock_critical, mock_error, mock_info):
        test_file = 'test_file.txt'
        with open(test_file, 'w') as f:
            f.write('Test data')
        size_instance = Size()
        result = size_instance.call(test_file)
        self.assertIn("File size:", result)
        mock_info.assert_called_once()
        os.remove(test_file)

    @patch('src.BreastCancer.utils.size.BrestCancer_info')
    @patch('src.BreastCancer.utils.size.BrestCancer_error')
    @patch('src.BreastCancer.utils.size.BrestCancer_critical')
    def test_call_file_not_found(self, mock_critical, mock_error, mock_info):
        size_instance = Size()
        with self.assertRaises(FileNotFoundError):
            size_instance.call('non_existent_file.txt')


    @patch('src.BreastCancer.utils.size.BrestCancer_info')
    @patch('src.BreastCancer.utils.size.BrestCancer_error')
    @patch('src.BreastCancer.utils.size.BrestCancer_critical')
    def test_call_permission_error(self, mock_critical, mock_error, mock_info):
        test_file = 'test_file.txt'
        with open(test_file, 'w') as f:
            f.write('Test data')
        with patch('os.path.getsize', side_effect=PermissionError("Simulated permission error")):
            size_instance = Size()
            with self.assertRaises(PermissionError):
                size_instance.call(test_file)
            mock_error.assert_called_once_with("PermissionError: Simulated permission error")
        os.remove(test_file)

    @patch('src.BreastCancer.utils.size.BrestCancer_info')
    @patch('src.BreastCancer.utils.size.BrestCancer_error')
    @patch('src.BreastCancer.utils.size.BrestCancer_critical')
    def test_call_os_error(self, mock_critical, mock_error, mock_info):
        test_file = 'test_file.txt'
        with open(test_file, 'w') as f:
            f.write('Test data')
        with patch('os.path.getsize', side_effect=OSError("Simulated OS error")):
            size_instance = Size()
            with self.assertRaises(OSError):
                size_instance.call(test_file)
            mock_critical.assert_called_once_with("OSError: Simulated OS error")
        os.remove(test_file)

if __name__ == '__main__':
    unittest.main()
