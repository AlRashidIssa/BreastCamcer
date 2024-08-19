import yaml
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BrestCancer import BrestCancer_error, BrestCancer_info, BrestCancer_warning


class IReadYaml(ABC):
    """
    Abstract base class for reading YAML configuration files.

    Defines the interface for reading a YAML file and returning its contents as a dictionary.
    """

    @abstractmethod
    def call(self, path_yaml: str) -> Dict[str, Any]:
        """
        Abstract method for reading a YAML file.

        Args:
            path_yaml (str): The path to the YAML file to be read.

        Returns:
            Dict[str, Any]: The contents of the YAML file as a dictionary.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass

class ReadYaml(IReadYaml):
    """
    Concrete implementation of the IReadYaml class.

    Provides functionality to read a YAML file and convert its contents into a dictionary.
    """

    def call(self, 
             path_yaml: str = "/home/alrashidissa/Desktop/BreastCancer/config.yaml"
             ) -> Dict[str, Any]:
        """
        Reads a YAML file and returns its contents as a dictionary.

        Args:
            path_yaml (str): The path to the YAML file to be read.

        Returns:
            Dict[str, Any]: The contents of the YAML file as a dictionary.

        Raises:
            FileNotFoundError: If the YAML file does not exist at the specified path.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        try:
            # Check if the file path is valid
            if not os.path.exists(path_yaml):
                BrestCancer_error(f"File not found: {path_yaml}")
                raise FileNotFoundError(f"YAML file not found: {path_yaml}")

            # Read and parse the YAML file
            with open(path_yaml, 'r') as file:
                config = yaml.safe_load(file)

            # Log information about successful reading
            BrestCancer_info(f"Successfully read YAML file: {path_yaml}")
            
            return config

        except FileNotFoundError as e:
            # Log and raise the exception for file not found
            BrestCancer_error(f"File not found error: {str(e)}")
            raise
        except yaml.YAMLError as e:
            # Log and raise the exception for YAML parsing errors
            BrestCancer_error(f"YAML parsing error: {str(e)}")
            raise
        except Exception as e:
            # Log and raise any other exceptions
            BrestCancer_error(f"An unexpected error occurred: {str(e)}")
            raise
