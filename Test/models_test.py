import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

from Test import test_info, test_warning, test_error, test_debug, test_critical
from src.BrestCancer.mdoels.algorithms  import (LogisticRegressionModel, SVCModel, DecisionTreeModel, 
                         RandomForestModel, KNeighborsModel, NaiveBayesModel, 
                         GradientBoostingModel)

class TestModelTraining(unittest.TestCase):
    @patch('joblib.dump')
    @patch('sklearn.linear_model.LogisticRegression.fit')
    def test_logistic_regression_training(self, mock_fit, mock_dump):
        model = LogisticRegressionModel()
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1])
        model_path = '/path/to/model.pkl'

        mock_fit.return_value = None
        mock_dump.return_value = None
        
        model.train(X, y, model_path)

        test_info("Testing Logistic Regression Model Training")
        mock_fit.assert_called_once_with(X, y)
        mock_dump.assert_called_once_with(model.model, model_path)
        test_info("Logistic Regression model trained and saved successfully")

    @patch('joblib.dump')
    @patch('sklearn.svm.SVC.fit')
    def test_svc_training(self, mock_fit, mock_dump):
        model = SVCModel()
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1])
        model_path = '/path/to/model.pkl'

        mock_fit.return_value = None
        mock_dump.return_value = None
        
        model.train(X, y, model_path)

        test_info("Testing SVC Model Training")
        mock_fit.assert_called_once_with(X, y)
        mock_dump.assert_called_once_with(model.model, model_path)
        test_info("SVC model trained and saved successfully")

    @patch('joblib.dump')
    @patch('sklearn.tree.DecisionTreeClassifier.fit')
    def test_decision_tree_training(self, mock_fit, mock_dump):
        model = DecisionTreeModel()
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1])
        model_path = '/path/to/model.pkl'

        mock_fit.return_value = None
        mock_dump.return_value = None
        
        model.train(X, y, model_path)

        test_info("Testing Decision Tree Model Training")
        mock_fit.assert_called_once_with(X, y)
        mock_dump.assert_called_once_with(model.model, model_path)
        test_info("Decision Tree model trained and saved successfully")

    @patch('joblib.dump')
    @patch('sklearn.ensemble.RandomForestClassifier.fit')
    def test_random_forest_training(self, mock_fit, mock_dump):
        model = RandomForestModel()
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1])
        model_path = '/path/to/model.pkl'

        mock_fit.return_value = None
        mock_dump.return_value = None
        
        model.train(X, y, model_path)

        test_info("Testing Random Forest Model Training")
        mock_fit.assert_called_once_with(X, y)
        mock_dump.assert_called_once_with(model.model, model_path)
        test_info("Random Forest model trained and saved successfully")

    @patch('joblib.dump')
    @patch('sklearn.neighbors.KNeighborsClassifier.fit')
    def test_knn_training(self, mock_fit, mock_dump):
        model = KNeighborsModel()
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1])
        model_path = '/path/to/model.pkl'

        mock_fit.return_value = None
        mock_dump.return_value = None
        
        model.train(X, y, model_path)

        test_info("Testing K-Nearest Neighbors Model Training")
        mock_fit.assert_called_once_with(X, y)
        mock_dump.assert_called_once_with(model.model, model_path)
        test_info("K-Nearest Neighbors model trained and saved successfully")

    @patch('joblib.dump')
    @patch('sklearn.naive_bayes.GaussianNB.fit')
    def test_naive_bayes_training(self, mock_fit, mock_dump):
        model = NaiveBayesModel()
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1])
        model_path = '/path/to/model.pkl'

        mock_fit.return_value = None
        mock_dump.return_value = None
        
        model.train(X, y, model_path)

        test_info("Testing Naive Bayes Model Training")
        mock_fit.assert_called_once_with(X, y)
        mock_dump.assert_called_once_with(model.model, model_path)
        test_info("Naive Bayes model trained and saved successfully")

    @patch('joblib.dump')
    @patch('sklearn.ensemble.GradientBoostingClassifier.fit')
    def test_gradient_boosting_training(self, mock_fit, mock_dump):
        model = GradientBoostingModel()
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1])
        model_path = '/path/to/model.pkl'

        mock_fit.return_value = None
        mock_dump.return_value = None
        
        model.train(X, y, model_path)

        test_info("Testing Gradient Boosting Model Training")
        mock_fit.assert_called_once_with(X, y)
        mock_dump.assert_called_once_with(model.model, model_path)
        test_info("Gradient Boosting model trained and saved successfully")

    @patch('joblib.dump')
    @patch('sklearn.linear_model.LogisticRegression.fit')
    def test_logistic_regression_training_error(self, mock_fit, mock_dump):
        model = LogisticRegressionModel()
        X = np.array([[1, 2], [3, 4]])
        y = pd.Series([0, 1])
        model_path = '/path/to/model.pkl'

        mock_fit.side_effect = Exception("Training error")

        with self.assertRaises(Exception):
            model.train(X, y, model_path)

        test_error("Error occurred during Logistic Regression model training")
        test_critical("Logistic Regression training raised an exception as expected")

if __name__ == '__main__':
    unittest.main()
