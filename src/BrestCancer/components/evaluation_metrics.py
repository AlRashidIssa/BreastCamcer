import pandas as pd
import sys
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef, log_loss)

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BrestCancer import BrestCancer_critical, BrestCancer_error, BrestCancer_info

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
            BrestCancer_info(f"Accurcy Model :{metrics['accuracy']}")
            BrestCancer_info(f"Precision Score :{metrics['precision']}")
            BrestCancer_info(f"Recall Score :{metrics['recall']}")
            BrestCancer_info(f"F1_Score :{metrics['f1']}")
            BrestCancer_info(f"ROC AUC :{metrics['roc_auc']}")
            BrestCancer_info(f"Confusion Matrix :{metrics['confusion_matrix']}")
            BrestCancer_info(f"Matthews Corrcoef :{metrics['matthews_corrcoef']}")
            BrestCancer_info(f"Log Loss :{metrics['log_loss']}")
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