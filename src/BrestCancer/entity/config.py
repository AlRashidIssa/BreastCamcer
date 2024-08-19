import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src/BrestCancer")
from utils.read_ymal import ReadYaml
from typing import Dict, Any

class CONFIG:
    """
    The Config class is responsible for loading configuration data from a YAML file and making it 
    easily accessible throughout the application. It reads the configuration file once and provides 
    attributes to access dataset, model, and preprocessing configurations.

    Attributes:
        dataset_config (Dict[str, Any]): Configuration related to the dataset, including paths and target column.
        model_config (Dict[str, Any]): Configuration related to the model, including paths to pre-trained models.
        preprocess_config (Dict[str, Any]): Configuration related to data preprocessing, such as columns to drop,
                                            scaling methods, and encodings.
        url (str): The URL of the dataset to be downloaded.
        zip_path (str): The path where the downloaded ZIP file will be saved.
        extract_to (str): The path where the dataset will be extracted.
        DFP (str): The path to the dataset file (CSV) after extraction.
        target (str): The target column in the dataset for prediction.
        model_path_s (str): The directory where pre-trained models are stored.
        path_model (str): The specific path to the saved model file.
        drop_columns (List[str]): List of columns to drop during preprocessing.
        drop_duplicates (bool): Whether to drop duplicate rows.
        drop_outliers (bool): Whether to drop outliers.
        missing_columns (List[str]): List of columns with missing data.
        missing_method (str): The method for handling missing data (e.g., "mean").
        scaler_method (str): The method for scaling data (e.g., "minmax").
        scaler_columns (List[str]): List of columns to scale.
        encoder_columns (List[str]): List of columns to encode.
        method (str): The method for encoding categorical variables.
        replce (bool): Whether to replace values in the dataset.
        value_replce (Dict[str, int]): Dictionary for value replacement (e.g., {'M': 1, 'B': 0}).
    """

    # Load the configuration from the YAML file
    config: Dict[str, Any] = ReadYaml().call()

    # Accessing dataset configuration
    dataset_config: Dict[str, Any] = config['dataset']
    model_config: Dict[str, Any] = config['model']
    preprocess_config: Dict[str, Any] = config['preprocess']

    # Dataset configuration attributes
    url: str = dataset_config["url"]
    zip_path: str = dataset_config["zip_path"]
    extract_to: str = dataset_config["extract_to"]
    DFP: str = dataset_config["DFP"]
    target: str = dataset_config["target"]

    # Model configuration attributes
    model_path_s: str = model_config["model_path_s"]
    path_model: str = model_config["path_model"]

    # Preprocessing configuration attributes
    drop_columns: list = preprocess_config["drop_columns"]
    drop_duplicates: bool = preprocess_config["drop_duplicates"]
    drop_outliers: bool = preprocess_config["drop_outliers"]
    missing_columns: list = preprocess_config["missing_columns"]
    missing_method: str = preprocess_config["missing_method"]
    scaler_method: str = preprocess_config["scaler_method"]
    scaler_columns: list = preprocess_config["scaler_columns"]
    encoder_columns: list = preprocess_config["encoder_columns"]
    method: str = preprocess_config["method"]
    replce: bool = preprocess_config["replce"]
    value_replce: Dict[str, int] = preprocess_config["value_replce"]