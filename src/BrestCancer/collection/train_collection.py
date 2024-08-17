"""
Script to train and evaluate various machine learning models on the Iris dataset.

Models trained:
- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Naive Bayes
- Gradient Boosting

The script initializes each model, trains it, saves it to a file, and then makes predictions on the test set.
"""

import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from BrestCancer import models_error, models_info
from BrestCancer.mdoels.algorithms import (DecisionTreeModel, GradientBoostingModel, KNeighborsModel, # type: ignore
                                           LogisticRegressionModel, NaiveBayesModel, RandomForestModel, SVCModel)

def train_and_evaluate_models(X_train, y_train, X_test, model_paths):
    """
    Train various models and evaluate them on the test set.

    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Test features.
    :param model_paths: Dictionary with model names and their corresponding file paths.
    """
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegressionModel(),
        'Support Vector Machine': SVCModel(),
        'Decision Tree': DecisionTreeModel(),
        'Random Forest': RandomForestModel(),
        'K-Nearest Neighbors': KNeighborsModel(),
        'Naive Bayes': NaiveBayesModel(),
        'Gradient Boosting': GradientBoostingModel(),
    }

    # Train and save models
    for name, model in models.items():
        models_info(f"Training {name}...")
        try:
            trained_model = model.train(X_train, y_train, model_paths[name])
            models_info(f"Training completed for {name}.")
        except Exception as e:
            models_error(f"Training failed for {name}: {e}")
            continue

        models_info(f"Making predictions with {name}...")
        try:
            predictions = trained_model.predict(X_test)
            models_info(f"{name} predictions: {predictions}")
        except Exception as e:
            models_error(f"Prediction failed for {name}: {e}")

if __name__ == "__main__":
    # Load dataset
    data = load_iris()
    X = data.data # type: ignore
    y = data.target # type: ignore

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model file paths
    model_paths = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Support Vector Machine': 'svc_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'K-Nearest Neighbors': 'knn_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Gradient Boosting': 'gradient_boosting_model.pkl',
    }

    # Train and evaluate models
    train_and_evaluate_models(X_train, y_train, X_test, model_paths)