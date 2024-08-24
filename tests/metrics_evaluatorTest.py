import unittest
import pandas as pd
import sys
from unittest.mock import patch, MagicMock

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src.BreastCancer.components.evaluation_metrics import MetricsEvaluator

class TestMetricsEvaluator(unittest.TestCase):

    def setUp(self):
        # Mock data for y_true and y_pred
        self.y_true = pd.Series([1, 0, 1, 1, 0, 1, 0])
        self.y_pred = pd.Series([1, 0, 0, 1, 0, 1, 1])

        # Instantiate the class
        self.evaluator = MetricsEvaluator()

    def test_metrics_evaluation(self):
        # Execute the call method
        metrics, plot_path = self.evaluator.call(self.y_true, self.y_pred)

        # Check if metrics are calculated correctly
        self.assertAlmostEqual(metrics['accuracy'], 0.714, places=2)
        self.assertAlmostEqual(metrics['precision'], 0.714, places=2)
        self.assertAlmostEqual(metrics['recall'], 0.714, places=2)
        self.assertAlmostEqual(metrics['f1'], 0.714, places=2)
        self.assertIsInstance(metrics['roc_auc'], float)
        self.assertIsInstance(metrics['matthews_corrcoef'], float)
        self.assertIsInstance(metrics['log_loss'], float)

        # Check if the plot path is correctly generated
        self.assertTrue(plot_path.endswith('confusion_matrix.png'))

    def test_invalid_input(self):
        # Test invalid input types
        with self.assertRaises(TypeError):
            self.evaluator.call([1, 0, 1], self.y_pred)

        # Test different lengths of y_true and y_pred
        with self.assertRaises(ValueError):
            self.evaluator.call(self.y_true, pd.Series([1, 0]))