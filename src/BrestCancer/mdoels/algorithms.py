import sys
import joblib # type: ignore
import numpy as np
import pandas as pd # type: ignore
from abc import ABC, abstractmethod
from typing import Any
from sklearn.base import BaseEstimator # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")


from BrestCancer import BrestCancer_info, BrestCancer_warning, BrestCancer_error, BrestCancer_debug, BrestCancer_critical

class IModel(ABC):
    """
    Interface for all machine learning models.

    Defines the contract for training using a machine learning model.
    """
    @abstractmethod
    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        """
        Train the model with the provided data and save it.

        :param X: Training features.
        :param y: Target labels.
        :param model_path_s: Path where the model will be saved.
        :return: The trained model.
        """
        pass

class LogisticRegressionModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = LogisticRegression()
            BrestCancer_info("Logistic Regression model initialized.")
        except Exception as e:
            BrestCancer_critical(f"Error initializing Logistic Regression model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        """
        Train the Logistic Regression model and save it.

        :param X: Training features.
        :param y: Target labels.
        :param model_path_s: Path where the model will be saved.
        :return: The trained Logistic Regression model.
        """
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            BrestCancer_info(f"Logistic Regression model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            BrestCancer_error(f"Error training Logistic Regression model: {e}")
            raise

class SVCModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = SVC()
            BrestCancer_info("Support Vector Machine model initialized.")
        except Exception as e:
            BrestCancer_critical(f"Error initializing Support Vector Machine model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        """
        Train the Support Vector Machine model and save it.

        :param X: Training features.
        :param y: Target labels.
        :param model_path_s: Path where the model will be saved.
        :return: The trained Support Vector Machine model.
        """
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            BrestCancer_info(f"Support Vector Machine model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            BrestCancer_error(f"Error training Support Vector Machine model: {e}")
            raise

class DecisionTreeModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = DecisionTreeClassifier()
            BrestCancer_info("Decision Tree model initialized.")
        except Exception as e:
            BrestCancer_critical(f"Error initializing Decision Tree model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        """
        Train the Decision Tree model and save it.

        :param X: Training features.
        :param y: Target labels.
        :param model_path_s: Path where the model will be saved.
        :return: The trained Decision Tree model.
        """
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            BrestCancer_info(f"Decision Tree model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            BrestCancer_error(f"Error training Decision Tree model: {e}")
            raise

class RandomForestModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = RandomForestClassifier()
            BrestCancer_info("Random Forest model initialized.")
        except Exception as e:
            BrestCancer_critical(f"Error initializing Random Forest model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        """
        Train the Random Forest model and save it.

        :param X: Training features.
        :param y: Target labels.
        :param model_path_s: Path where the model will be saved.
        :return: The trained Random Forest model.
        """
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            BrestCancer_info(f"Random Forest model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            BrestCancer_error(f"Error training Random Forest model: {e}")
            raise

class KNeighborsModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = KNeighborsClassifier()
            BrestCancer_info("K-Nearest Neighbors model initialized.")
        except Exception as e:
            BrestCancer_critical(f"Error initializing K-Nearest Neighbors model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        """
        Train the K-Nearest Neighbors model and save it.

        :param X: Training features.
        :param y: Target labels.
        :param model_path_s: Path where the model will be saved.
        :return: The trained K-Nearest Neighbors model.
        """
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            BrestCancer_info(f"K-Nearest Neighbors model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            BrestCancer_error(f"Error training K-Nearest Neighbors model: {e}")
            raise

class NaiveBayesModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = GaussianNB()
            BrestCancer_info("Naive Bayes model initialized.")
        except Exception as e:
            BrestCancer_critical(f"Error initializing Naive Bayes model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        """
        Train the Naive Bayes model and save it.

        :param X: Training features.
        :param y: Target labels.
        :param model_path_s: Path where the model will be saved.
        :return: The trained Naive Bayes model.
        """
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            BrestCancer_info(f"Naive Bayes model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            BrestCancer_error(f"Error training Naive Bayes model: {e}")
            raise

class GradientBoostingModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = GradientBoostingClassifier()
            BrestCancer_info("Gradient Boosting model initialized.")
        except Exception as e:
            BrestCancer_critical(f"Error initializing Gradient Boosting model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        """
        Train the Gradient Boosting model and save it.

        :param X: Training features.
        :param y: Target labels.
        :param model_path_s: Path where the model will be saved.
        :return: The trained Gradient Boosting model.
        """
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, f"{model_path_s}/GradientBoostingModel.pkl")
            BrestCancer_info(f"Gradient Boosting model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            BrestCancer_error(f"Error training Gradient Boosting model: {e}")
            raise
