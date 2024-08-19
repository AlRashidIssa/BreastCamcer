import pandas as pd
import sys
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef, log_loss)

sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src import BrestCancer_critical, BrestCancer_error, BrestCancer_info, BrestCancer_warning

class IMetricsEvaluator(ABC):
    """
    Abstract base class for evaluating metrics.

    This class defines the interface for evaluating various metrics for classification tasks.
    It requires implementing classes to provide their own `call` method to calculate and return metrics.
    """

    @abstractmethod
    def call(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Union[float, str, list]]:
        """
        Abstract method for calculating metrics.

        Args:
            y_true (pd.Series): The true labels.
            y_pred (pd.Series): The predicted labels.

        Returns:
            Dict[str, Union[float, str, list]]: A dictionary containing metric names as keys and their corresponding values.
        """
        pass

class MetricsEvaluator(IMetricsEvaluator):
    """
    Concrete implementation of the IMetricsEvaluator interface for calculating classification metrics.

    This class implements the `call` method to compute a variety of classification metrics such as accuracy, precision, recall,
    F1 score, ROC AUC score, confusion matrix, Matthews correlation coefficient, and log loss.
    """

    def call(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Union[float, str, list]]:
        """
        Calculate and return a dictionary of classification metrics.

        Args:
            y_true (pd.Series): The true labels.
            y_pred (pd.Series): The predicted labels.

        Returns:
            Dict[str, Union[float, str, list]]: A dictionary containing calculated metrics such as accuracy, precision, recall, F1 score,
                                                 ROC AUC score, confusion matrix, Matthews correlation coefficient, and log loss.

        Raises:
            ValueError: If `y_true` and `y_pred` do not have the same length.
            TypeError: If the input arguments are not pandas Series.
            Exception: For any other unforeseen errors during metric computation.
        """
        if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
            BrestCancer_error("Inputs must be pandas Series.")
            raise TypeError("Inputs must be pandas Series.")

        if len(y_true) != len(y_pred):
            BrestCancer_error("Length of y_true and y_pred must be the same.")
            raise ValueError("Length of y_true and y_pred must be the same.")

        metrics = {}

        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for easier JSON serialization
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
            metrics['log_loss'] = log_loss(y_true, y_pred)

            BrestCancer_info("Metrics calculated successfully.")
        except ValueError as ve:
            BrestCancer_error(f"Value error occurred: {ve}")
            raise
        except TypeError as te:
            BrestCancer_error(f"Type error occurred: {te}")
            raise
        except Exception as e:
            BrestCancer_critical(f"An unexpected error occurred: {e}")
            raise

        return metrics
    

import pandas as pd
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=20, random_state=42)
y_true = pd.Series(y)  # True labels

# Simulate predictions (for demonstration purposes)
y_pred = pd.Series(y)  # In a real scenario, these would be the predictions from your model

# Instantiate the MetricsEvaluator
evaluator = MetricsEvaluator()

try:
    # Calculate metrics
    metrics = evaluator.call(y_true, y_pred)
    
    # Print the metrics
    print("Metrics calculated:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

except ValueError as ve:
    print(f"ValueError: {ve}")
except TypeError as te:
    print(f"TypeError: {te}")
except Exception as e:
    print(f"Unexpected error: {e}")

