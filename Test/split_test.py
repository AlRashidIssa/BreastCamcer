# import unittest
# import pandas as pd
# import sys
# from unittest.mock import patch, MagicMock

# sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

# from src.BrestCancer.ingestdata.split import Split 
# from Test import test_info, test_warning, test_error, test_debug, test_critical

# class TestSplit(unittest.TestCase):

#     @patch('src.BrestCancer.ingestdata.split.Ingest')
#     @patch('src.BrestCancer.ingestdata.split.ingest_info')
#     @patch('src.BrestCancer.ingestdata.split.ingest_warning')
#     @patch('src.BrestCancer.ingestdata.split.ingest_error')
#     @patch('src.BrestCancer.ingestdata.split.ingest_critical')
#     def test_call_success(self, mock_ingest_critical, mock_ingest_error, mock_ingest_warning, mock_ingest_info, mock_Ingest):
#         # Setup
#         test_csv_path = 'test_data/breast-cancer.csv'
#         mock_Ingest.return_value.call.return_value = pd.read_csv(test_csv_path)
        
#         splitter = Split()

#         # Act
#         X_train, X_test, y_train, y_test = splitter.call('diagnosis', test_csv_path)

#         # Assert
#         self.assertGreaterEqual(X_train.shape[0], 1)  # Adjust based on actual test CSV data
#         self.assertGreaterEqual(X_test.shape[0], 1)   # Adjust based on actual test CSV data
#         self.assertEqual(len(y_train), len(X_train))
#         self.assertEqual(len(y_test), len(X_test))

#         # Check logging
#         mock_ingest_info.assert_called_with(f"Reading data from {test_csv_path}")
#         mock_ingest_debug.assert_called_with("Data split into training and testing sets.")


#     @patch('src.BrestCancer.ingestdata.split.Ingest')
#     @patch('src.BrestCancer.ingestdata.split.ingest_info')
#     @patch('src.BrestCancer.ingestdata.split.ingest_warning')
#     @patch('src.BrestCancer.ingestdata.split.ingest_error')
#     @patch('src.BrestCancer.ingestdata.split.ingest_critical')
#     def test_empty_dataset(self, mock_ingest_critical, mock_ingest_error, mock_ingest_warning, mock_ingest_info, mock_Ingest):
#         # Setup
#         empty_csv_path = 'test_data/empty.csv'
#         mock_Ingest.return_value.call.return_value = pd.read_csv(empty_csv_path)
        
#         splitter = Split()

#         with self.assertRaises(ValueError):
#             splitter.call('diagnosis', empty_csv_path)

#         # Check logging
#         mock_ingest_warning.assert_called_with("The dataset is empty.")
    
#     @patch('src.BrestCancer.ingestdata.split.Ingest')
#     @patch('src.BrestCancer.ingestdata.split.ingest_info')
#     @patch('src.BrestCancer.ingestdata.split.ingest_warning')
#     @patch('src.BrestCancer.ingestdata.split.ingest_error')
#     @patch('src.BrestCancer.ingestdata.split.ingest_critical')
#     def test_target_column_not_found(self, mock_ingest_critical, mock_ingest_error, mock_ingest_warning, mock_ingest_info, mock_Ingest):
#         # Setup
#         target_not_found_csv_path = 'test_data/target_not_found.csv'
#         mock_Ingest.return_value.call.return_value = pd.read_csv(target_not_found_csv_path)
        
#         splitter = Split()

#         with self.assertRaises(ValueError):
#             splitter.call('target', target_not_found_csv_path)

#         # Check logging
#         mock_ingest_error.assert_called_with("Target column 'target' not found in the dataset.")
    
#     @patch('src.BrestCancer.ingestdata.split.Ingest')
#     @patch('src.BrestCancer.ingestdata.split.ingest_info')
#     @patch('src.BrestCancer.ingestdata.split.ingest_warning')
#     @patch('src.BrestCancer.ingestdata.split.ingest_error')
#     @patch('src.BrestCancer.ingestdata.split.ingest_critical')
#     def test_unexpected_error(self, mock_ingest_critical, mock_ingest_error, mock_ingest_warning, mock_ingest_info, mock_Ingest):
#         # Setup
#         mock_Ingest.side_effect = Exception("Unexpected error")
        
#         splitter = Split()

#         with self.assertRaises(RuntimeError):
#             splitter.call('diagnosis', 'test_data/breast-cancer.csv')

#         # Check logging
#         mock_ingest_critical.assert_called_with("An unexpected error occurred: Unexpected error")

# if __name__ == '__main__':
#     unittest.main()
