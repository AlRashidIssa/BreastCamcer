import os
import sys
from typing import Dict, Any, List
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from src.BreastCancer.utils.read_ymal import ReadYaml

class CONFIG:
    """
    The CONFIG class is responsible for loading configuration data from a YAML file and making it 
    easily accessible throughout the application. It reads the configuration file once and provides 
    attributes to access dataset, model, and preprocessing configurations.
    """
    def __init__(self, yaml_path) -> None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        # Load the configuration from the YAML file
        self.config: Dict[str, Any] = ReadYaml().call(path_yaml=yaml_path)
        
        # Accessing dataset configuration
        self.dataset_config: Dict[str, Any] = self.config['dataset']
        self.model_config: Dict[str, Any] = self.config['model']

        self.preprocess_config: Dict[str, Any] = self.config['preprocess']
        # Dataset configuration attributes
        self.url: str = self.dataset_config["url"]
        self.name_dataset: str = self.dataset_config["name_dataset"]
        self.download: str = self.dataset_config["download"]
        self.zip_path: str = self.dataset_config["zip_path"]
        self.extract_to: str = self.dataset_config["extract_to"]
        self.DFP: str = self.dataset_config["DFP"]
        self.target: str = self.dataset_config["target"]
        # Model configuration attributes
        self.name_model: str = self.model_config["name_model"]
        self.model_path_s: str = self.model_config["model_path_s"]
        self.path_model: str = self.model_config["path_model"]

        # Preprocessing configuration attributes
        self.handling = self.preprocess_config["handling"]
        self.handl_missing: bool = self.preprocess_config["handl_missing"]
        self.fill_na: bool = self.preprocess_config["fill_na"]
        self.fill_value: float = self.preprocess_config["fill_value"]
        self.drop_columns: List[str] = self.preprocess_config["drop_columns"]
        self.drop_duplicates: bool = self.preprocess_config["drop_duplicates"]
        self.drop_outliers: bool = self.preprocess_config["drop_outliers"]
        self.missing_column: str = self.preprocess_config["missing_columns"]
        self.missing_method: str = self.preprocess_config["missing_method"]
        self.scaler_method: str = self.preprocess_config["scaler_method"]
        self.scaler_columns: List[str] = self.preprocess_config["scaler_columns"]
        self.encoder_columns: List[str] = self.preprocess_config["encoder_columns"]
        self.method_encoder: str = self.preprocess_config["method_encoder"]
        self.replce: bool = self.preprocess_config["replce"]
        self.value_replce: Dict[str, int] = self.preprocess_config["value_replce"]
        self.plotes_pathes: str = self.preprocess_config["plotes_pathes"]
