import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add the project directory to the system path
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

# Import the Ingest class and logging functions
from src.BrestCancer.ingestdata.ingesting import Ingest
from Test import test_info, test_warning, test_error, test_debug, test_critical

class TestIngest(unittest.TestCase):
    """
    Unit tests for the Ingest class.
    """

    @patch('src.BrestCancer.ingestdata.ingesting.pd.read_csv')
    @patch('src.BrestCancer.ingestdata.ingesting.os.path.exists')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_info')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_warning')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_error')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_debug')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_critical')
    def test_call_success(self, mock_critical, mock_debug, mock_error, mock_warning, mock_info, mock_exists, mock_read_csv):
        """
        Test the `call` method with a valid file path.
        """
        # Arrange
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        # Create an instance of the Ingest class
        ingest = Ingest()
        
        # Act
        df = ingest.call('valid_path.csv')
        
        # Assert
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        mock_debug.assert_called_once_with("Starting Ingest Data From valid_path.csv")
        mock_info.assert_called_once_with("Successfully ingested data from valid_path.csv")

    @patch('src.BrestCancer.ingestdata.ingesting.os.path.exists')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_error')
    def test_call_file_not_found(self, mock_error, mock_exists):
        """
        Test the `call` method with a non-existent file path.
        """
        # Arrange
        mock_exists.return_value = False
        ingest = Ingest()
        
        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            ingest.call('invalid_path.csv')
        mock_error.assert_called_once_with("File Ingest Data, Invalid Path: invalid_path.csv")

    @patch('src.BrestCancer.ingestdata.ingesting.pd.read_csv')
    @patch('src.BrestCancer.ingestdata.ingesting.os.path.exists')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_error')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_warning')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_debug')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_info')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_critical')
    def test_call_empty_file(self, mock_critical, mock_info, mock_debug, mock_warning, mock_error, mock_exists, mock_read_csv):
        """
        Test the `call` method with an empty file.
        """
        # Arrange
        mock_exists.return_value = True
        mock_read_csv.side_effect = pd.errors.EmptyDataError
        ingest = Ingest()
        
        # Act & Assert
        with self.assertRaises(ValueError):
            ingest.call('empty_file.csv')
        mock_warning.assert_called_once_with("The file empty_file.csv is empty.")

    @patch('src.BrestCancer.ingestdata.ingesting.pd.read_csv')
    @patch('src.BrestCancer.ingestdata.ingesting.os.path.exists')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_error')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_warning')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_debug')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_info')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_critical')
    def test_call_parser_error(self, mock_critical, mock_info, mock_debug, mock_warning, mock_error, mock_exists, mock_read_csv):
        """
        Test the `call` method with a parsing error.
        """
        # Arrange
        mock_exists.return_value = True
        mock_read_csv.side_effect = pd.errors.ParserError
        ingest = Ingest()
        
        # Act & Assert
        with self.assertRaises(ValueError):
            ingest.call('parser_error_file.csv')  # Ensure to call the `call` method of `Ingest`
        mock_error.assert_called_once_with("Error parsing the file parser_error_file.csv.")
    

    @patch('src.BrestCancer.ingestdata.ingesting.pd.read_csv')
    @patch('src.BrestCancer.ingestdata.ingesting.os.path.exists')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_error')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_warning')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_debug')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_info')
    @patch('src.BrestCancer.ingestdata.ingesting.ingest_critical')
    def test_call_unexpected_error(self, mock_critical, mock_info, mock_debug, mock_warning, mock_error, mock_exists, mock_read_csv):
        """
        Test the `call` method with an unexpected error.
        """
        # Arrange
        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Unexpected error")
        ingest = Ingest()
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            ingest.call('unexpected_error_file.csv')
        mock_critical.assert_called_once_with("An unexpected error occurred: Unexpected error")

if __name__ == '__main__':
    unittest.main()
