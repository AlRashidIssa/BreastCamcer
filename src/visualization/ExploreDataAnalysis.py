import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any
from io import StringIO

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from src import BrestCancer_info, BrestCancer_debug, BrestCancer_warning

# Interface for Data Operations
class DataOperation(ABC):
    """
    Abstract base class for data operations.

    Classes inheriting from DataOperation must implement the execute method.
    This is used to enforce a consistent interface for different data operations.
    """

    @abstractmethod
    def execute(self):
        """
        Execute the operation.
        
        Must be implemented by any subclass.
        """
        pass

# Dataset Loader
class DatasetLoader(DataOperation):
    """
    Class responsible for loading a dataset from a given CSV file.

    Attributes:
        filepath (str): Path to the CSV file.
        data (pd.DataFrame): DataFrame containing the loaded data.
    """

    def __init__(self, filepath: str):
        """
        Initialize DatasetLoader with the path to the dataset.

        Args:
            filepath (str): Path to the CSV file.
        """
        self.filepath = filepath
        self.data = None

    def execute(self) -> pd.DataFrame:
        """
        Load the dataset from the CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            if self.data.empty:
                raise ValueError(f"Loaded dataset is empty from {self.filepath}")
            BrestCancer_info(f"Dataset loaded successfully from {self.filepath}")
            return self.data
        except Exception as e:
            BrestCancer_warning(f"Failed to load dataset from {self.filepath}: {e}")
            raise

# Data Inspector
class DataInspector:
    """
    Class responsible for inspecting a dataset for basic information,
    such as shape, missing values, duplicates, and summary statistics.

    Attributes:
        data (pd.DataFrame): DataFrame to be inspected.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataInspector with the dataset.

        Args:
            data (pd.DataFrame): DataFrame to be inspected.
        """
        self.data = data

    def execute(self) -> Dict[str, Any]:
        """
        Perform basic inspection of the dataset and return the results as a dictionary.
        
        Returns:
            Dict[str, Any]: A dictionary containing dataset inspection results.
        """
        BrestCancer_info("Inspecting dataset")
        BrestCancer_debug(f"Shape of Dataset :{self.data.head()}")
        BrestCancer_debug(f"First 5 Rows of the Dataset:\n{self.data.head()}")
        buffer = StringIO()
        self.data.info(buf=buffer)
        data_info = buffer.getvalue()
        BrestCancer_debug(f"Data Info:\n{data_info}")
        BrestCancer_debug(f"Statistical Summary:\n{self.data.describe()}")
        BrestCancer_debug(f"Missing Values in Each Column:\n{self.data.isnull().sum()}")
        BrestCancer_debug(f"Number of Duplicate Rows: {self.data.duplicated().sum()}")
        data_inspected = {
            "First 5 Rows of the Dataset": self.data.head(),
            "Missing Values in Each Column": self.data.isnull().sum(),
            "Number of Duplicate Rows": self.data.duplicated().sum(),
            "Data Information": data_info,
            "Statistical Summary": self.data.describe()
        }
        return data_inspected

# Base class for Data Visualizations
class DataVisualizer(ABC):
    """
    Abstract base class for data visualizations.

    This class provides a structure for visualizing data and saving plots to files.
    """

    def __init__(self, data: pd.DataFrame, output_dir: str = "/home/alrashidissa/Desktop/BreastCancer/Plots"):
        """
        Initialize DataVisualizer with the dataset and output directory.

        Args:
            data (pd.DataFrame): DataFrame containing the data to be visualized.
            output_dir (str): Directory where the plots will be saved. Default is 'plots'.
        """
        self.data = data
        self.output_dir = output_dir

        # Create the output directory if it does not exist
        if not os.path.exists(self.output_dir):
            BrestCancer_warning(f"No such Directory Exists: {self.output_dir}")
            os.makedirs(self.output_dir)
            BrestCancer_info(f"Successfully Created new Directory. Dir Path: {self.output_dir}")

    @abstractmethod
    def visualize(self):
        """
        Abstract method to be implemented by subclasses for specific visualizations.
        """
        pass

    def save_plot(self, filename: str):
        """
        Save the current plot to a file in the specified output directory.

        Args:
            filename (str): Name of the file to save the plot as.
        """
        try:
            file_path = os.path.join(self.output_dir, filename)
            plt.savefig(file_path)
            BrestCancer_info(f"Plot saved as {filename} in {self.output_dir}")
        except Exception as e:
            BrestCancer_warning(f"Failed to save plot {filename}: {e}")
        finally:
            plt.close()

# Visualization for Target Variable
class TargetVariableVisualizer(DataVisualizer):
    """
    Class responsible for visualizing the distribution of the target variable.

    This typically includes creating a count plot of the target variable.
    """

    def visualize(self):
        """
        Create and save a count plot of the target variable distribution.
        """
        BrestCancer_info("Visualizing distribution of target variable")
        sns.countplot(x='diagnosis', data=self.data)
        plt.title('Distribution of Diagnosis (Target Variable)')
        self.save_plot('target_variable_distribution.png')

# Visualization for Feature Distribution (Univariate Analysis)
class FeatureDistributionVisualizer(DataVisualizer):
    """
    Class responsible for visualizing the distribution of selected features.

    Attributes:
        features (list): List of feature names to visualize.
    """

    def __init__(self, data: pd.DataFrame, features: List[str], output_dir: str = "/home/alrashidissa/Desktop/BreastCancer/Plots"):
        """
        Initialize FeatureDistributionVisualizer with the dataset and features.

        Args:
            data (pd.DataFrame): DataFrame containing the data to be visualized.
            features (list): List of feature names to visualize.
            output_dir (str): Directory where the plots will be saved. Default is 'plots'.
        """
        super().__init__(data, output_dir)
        self.features = features

    def visualize(self):
        """
        Create and save histograms of the selected features.
        """
        BrestCancer_info("Visualizing distribution of selected features")
        self.data[self.features].hist(bins=15, figsize=(15, 10), layout=(2, 3))
        plt.suptitle('Distribution of Selected Features')
        self.save_plot('feature_distribution.png')

# Visualization for Correlation Matrix
class CorrelationMatrixVisualizer(DataVisualizer):
    """
    Class responsible for visualizing the correlation matrix of numerical features.
    """

    def visualize(self):
        """
        Create and save a heatmap of the correlation matrix.
        """
        BrestCancer_info("Visualizing correlation matrix")
        numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numerical_data.corr()
        plt.figure(figsize=(18, 15))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix')
        self.save_plot('correlation_matrix.png')

# Visualization for Pairplot
class PairplotVisualizer(DataVisualizer):
    """
    Class responsible for creating pairplots for selected features.

    Attributes:
        features (list): List of feature names to visualize.
        hue (str): The column name to use for color encoding in the pairplot.
    """

    def __init__(self, data: pd.DataFrame, features: List[str], hue: str, output_dir: str = "/home/alrashidissa/Desktop/BreastCancer/Plots"):
        """
        Initialize PairplotVisualizer with the dataset, features, and hue.

        Args:
            data (pd.DataFrame): DataFrame containing the data to be visualized.
            features (list): List of feature names to visualize.
            hue (str): The column name to use for color encoding in the pairplot.
            output_dir (str): Directory where the plots will be saved. Default is 'plots'.
        """
        super().__init__(data, output_dir)
        self.features = features
        self.hue = hue

    def visualize(self):
        """
        Create and save a pairplot for the selected features.
        """
        BrestCancer_info("Visualizing pairplot of selected features")
        sns.pairplot(self.data[self.features], hue=self.hue)
        plt.title('Pairplot of Selected Features')
        self.save_plot('pairplot.png')

# Visualization for Boxplot
class BoxplotVisualizer(DataVisualizer):
    """
    Class responsible for creating boxplots for selected features with respect to a target variable.

    Attributes:
        features (list): List of feature names to visualize.
    """

    def __init__(self, data: pd.DataFrame, features: List[str], output_dir: str = "/home/alrashidissa/Desktop/BreastCancer/Plots"):
        """
        Initialize BoxplotVisualizer with the dataset and features.

        Args:
            data (pd.DataFrame): DataFrame containing the data to be visualized.
            features (list): List of feature names to visualize.
            output_dir (str): Directory where the plots will be saved. Default is 'plots'.
        """
        super().__init__(data, output_dir)
        self.features = features

    def visualize(self):
        """
        Create and save boxplots of the selected features with respect to the diagnosis.
        """
        BrestCancer_info("Visualizing boxplots of selected features")
        plt.figure(figsize=(18, 10))
        for i, feature in enumerate(self.features):
            plt.subplot(2, 3, i+1)
            sns.boxplot(x='diagnosis', y=feature, data=self.data)
            plt.title(f'{feature} vs Diagnosis')
        plt.tight_layout()
        self.save_plot('boxplot.png')

# Main Analyzer Class
class BreastCancerAnalyzer:
    """
    Main class for analyzing the Breast Cancer dataset.

    This class integrates various components (data loading, inspection, visualization)
    to perform a comprehensive analysis.

    Attributes:
        loader (DatasetLoader): Loader for the dataset.
        data (pd.DataFrame): DataFrame containing the loaded data.
    """

    def __init__(self, filepath: str, output_dir: str = "/home/alrashidissa/Desktop/BreastCancer/Plots"):
        """
        Initialize BreastCancerAnalyzer with the path to the dataset and output directory.

        Args:
            filepath (str): Path to the CSV file containing the dataset.
            output_dir (str): Directory where the plots will be saved.
        """
        self.loader = DatasetLoader(filepath)
        self.output_dir = output_dir

    def load_data(self):
        """
        Load the dataset using the DatasetLoader.
        """
        BrestCancer_info("Loading data")
        self.data = self.loader.execute()

    def inspect_data(self) -> Dict[str, Any]:
        """
        Inspect the dataset using the DataInspector.
        """
        BrestCancer_info("Inspecting data")
        inspector = DataInspector(self.data)
        data_inspected = inspector.execute()
        return data_inspected

    def analyze(self):
        """
        Perform a comprehensive analysis of the dataset, including visualization of target variable,
        feature distributions, correlation matrix, pairplots, and boxplots.
        """
        BrestCancer_info("Starting analysis")

        # Target Variable Analysis
        BrestCancer_info("Analyzing target variable")
        target_visualizer = TargetVariableVisualizer(self.data, self.output_dir)
        target_visualizer.visualize()

        # Univariate Analysis
        BrestCancer_info("Analyzing feature distributions")
        features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
        feature_visualizer = FeatureDistributionVisualizer(self.data, features, self.output_dir)
        feature_visualizer.visualize()

        # Correlation Matrix
        BrestCancer_info("Analyzing correlation matrix")
        correlation_visualizer = CorrelationMatrixVisualizer(self.data, self.output_dir)
        correlation_visualizer.visualize()

        # Pairplot Analysis
        BrestCancer_info("Analyzing pairplot")
        selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
        pairplot_visualizer = PairplotVisualizer(self.data, selected_features, hue='diagnosis', output_dir=self.output_dir)
        pairplot_visualizer.visualize()

        # Boxplot Analysis
        BrestCancer_info("Analyzing boxplots")
        boxplot_visualizer = BoxplotVisualizer(self.data, features, self.output_dir)
        boxplot_visualizer.visualize()

# Example Usage
if __name__ == "__main__":
    output_dir = "/home/alrashidissa/Desktop/BreastCancer/Plots"
    analyzer = BreastCancerAnalyzer("/home/alrashidissa/Desktop/BreastCancer/Dataset/extract/breast-cancer.csv", output_dir)
    analyzer.load_data()
    data_inspected = analyzer.inspect_data()
    analyzer.analyze()
