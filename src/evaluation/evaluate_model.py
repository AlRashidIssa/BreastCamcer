import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef, log_loss)

sys.path.append("/home/alrashid/Desktop/BreastCancer")
from src.utils.logging import info, error, critical

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
    def call(self, y_true: pd.Series, y_pred: pd.Series) -> Tuple[Dict[str, Union[float, str, list]], str]:
        """
        Calculate and return a dictionary of classification metrics and the path to the saved confusion matrix plot.

        Args:
            y_true (pd.Series): The true labels.
            y_pred (pd.Series): The predicted labels.

        Returns:
            Dict[str, Union[float, str, list]]: A dictionary containing calculated metrics such as accuracy, precision, recall, F1 score,
                                                 ROC AUC score, confusion matrix, Matthews correlation coefficient, and log loss.
            str: Path to the saved confusion matrix plot.

        Raises:
            ValueError: If `y_true` and `y_pred` do not have the same length.
            TypeError: If the input arguments are not pandas Series.
            Exception: For any other unforeseen errors during metric computation.
        """
        if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
            error("Inputs must be pandas Series.")
            raise TypeError("Inputs must be pandas Series.")

        if len(y_true) != len(y_pred):
            error("Length of y_true and y_pred must be the same.")
            raise ValueError("Length of y_true and y_pred must be the same.")

        metrics = {}
        confusion_matrix_pram = None
        plot_path = ''

        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            confusion_matrix_pram = confusion_matrix(y_true, y_pred)
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
            metrics['log_loss'] = log_loss(y_true, y_pred)
            
            # Log metrics
            info(f"Accuracy Model: {metrics['accuracy']}")
            info(f"Precision Score: {metrics['precision']}")
            info(f"Recall Score: {metrics['recall']}")
            info(f"F1 Score: {metrics['f1']}")
            info(f"ROC AUC: {metrics['roc_auc']}")
            info(f"Matthews Corrcoef: {metrics['matthews_corrcoef']}")
            info(f"Log Loss: {metrics['log_loss']}")
            info("Metrics calculated successfully.")
            
            path_plot = "/home/alrashid/Desktop/BreastCancer/notebooks/prototyping"
            # Plot and save confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(confusion_matrix_pram, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title('Confusion Matrix')
            
            # Ensure the plots directory exists
            os.makedirs(path_plot, exist_ok=True)
            
            # Define the full path for the plot
            plot_filename = 'confusion_matrix.png'
            plot_path = os.path.join(path_plot, plot_filename)
            
            # Save the plot
            fig.savefig(plot_path)
            plt.close(fig)
        except ValueError as ve:
            error(f"Value error occurred: {ve}")
            raise
        except TypeError as te:
            error(f"Type error occurred: {te}")
            raise
        except Exception as e:
            critical(f"An unexpected error occurred: {e}")
            raise

        return metrics, plot_path