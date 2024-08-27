import sys
import joblib  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from abc import ABC, abstractmethod
from typing import Any
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.naive_bayes import GaussianNB  # type: ignore

sys.path.append("/app")
from src.utils.logging import info, error,  critical

# Define the IModel interface
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

# Implementations of the different models
class LogisticRegressionModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = LogisticRegression()
            info("Logistic Regression model initialized.")
        except Exception as e:
            critical(f"Error initializing Logistic Regression model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            info(f"Logistic Regression model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            error(f"Error training Logistic Regression model: {e}")
            raise

class SVCModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = SVC()
            info("Support Vector Machine model initialized.")
        except Exception as e:
            critical(f"Error initializing Support Vector Machine model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            info(f"Support Vector Machine model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            error(f"Error training Support Vector Machine model: {e}")
            raise

class DecisionTreeModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = DecisionTreeClassifier()
            info("Decision Tree model initialized.")
        except Exception as e:
            critical(f"Error initializing Decision Tree model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            info(f"Decision Tree model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            error(f"Error training Decision Tree model: {e}")
            raise

class RandomForestModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = RandomForestClassifier()
            info("Random Forest model initialized.")
        except Exception as e:
            critical(f"Error initializing Random Forest model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            info(f"Random Forest model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            error(f"Error training Random Forest model: {e}")
            raise

class KNeighborsModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = KNeighborsClassifier()
            info("K-Nearest Neighbors model initialized.")
        except Exception as e:
            critical(f"Error initializing K-Nearest Neighbors model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            info(f"K-Nearest Neighbors model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            error(f"Error training K-Nearest Neighbors model: {e}")
            raise

class NaiveBayesModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = GaussianNB()
            info("Naive Bayes model initialized.")
        except Exception as e:
            critical(f"Error initializing Naive Bayes model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, model_path_s)
            info(f"Naive Bayes model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            error(f"Error training Naive Bayes model: {e}")
            raise

class GradientBoostingModel(IModel):
    def __init__(self) -> None:
        try:
            self.model = GradientBoostingClassifier()
            info("Gradient Boosting model initialized.")
        except Exception as e:
            critical(f"Error initializing Gradient Boosting model: {e}")
            raise

    def train(self, X: np.ndarray, y: pd.Series, model_path_s: str) -> BaseEstimator:
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, f"{model_path_s}/GradientBoostingModel.pkl")
            info(f"Gradient Boosting model trained and saved to {model_path_s}.")
            return self.model
        except Exception as e:
            error(f"Error training Gradient Boosting model: {e}")
            raise

# Interface for choosing a model
class IChooseModel(ABC):
    """
    Interface for selecting a machine learning model.

    Defines the contract for selecting and returning a machine learning model.
    """
    @abstractmethod
    def call(self, name_model: str = "GBM") -> IModel:
        """
        Select and return a machine learning model based on the provided name.

        :param name_model: The name of the model to select.
        :return: The selected model instance.
        """
        pass

# Implementation for choosing a model
class ChooseModel(IChooseModel):
    """
    Implementation of the IChooseModel interface.

    Provides the functionality to select and return a machine learning model based on the provided name.
    """
    def call(self, name_model: str = "GBM") -> IModel:
        """
        Select and return a machine learning model based on the provided name.

        :param name_model: The name of the model to select. Default is "GBM".
        :return: The selected model instance.
        :raises ValueError: If the provided model name is not recognized.
        :support Machine learining Model: [LogisticRegression, SVM, DecisionTree, RandomForest,
                                           KNN, NaiveBayes, GBM]
        """
        try:
            if name_model == "LogisticRegressionModel":
                info("Logistic Regression model selected.")
                return LogisticRegressionModel()
            elif name_model == "SVCModel":
                info("Support Vector Machine model selected.")
                return SVCModel()
            elif name_model == "DecisionTreeModel":
                info("Decision Tree model selected.")
                return DecisionTreeModel()
            elif name_model == "RandomForestModel":
                info("Random Forest model selected.")
                return RandomForestModel()
            elif name_model == "KNeighborsModel":
                info("K-Nearest Neighbors model selected.")
                return KNeighborsModel()
            elif name_model == "NaiveBayesModel":
                info("Naive Bayes model selected.")
                return NaiveBayesModel()
            elif name_model == "GradientBoostingModel":
                info("Gradient Boosting model selected.")
                return GradientBoostingModel()
            else:
                error(f"Unknown model: {name_model}")
                raise ValueError(f"Unknown model: {name_model}")
        except Exception as e:
            critical(f"Error selecting model: {e}")
            raise