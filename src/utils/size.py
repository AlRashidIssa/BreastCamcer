from abc import ABC, abstractmethod
import os
import numpy as np # type: ignore

import sys
sys.path.append("/home/alrashid/Desktop/BreastCancer")
from src.utils.logging import critical, error, info


class ISize(ABC):
    """
    Abstract interface for getting the size of a file.
    """
    @abstractmethod
    def call(self, path) -> str:
        """
        Abstract method to get the file size.

        :param path: Path to the file.
        :return: File size in MB as a string.
        """
        pass


class Size(ISize):
    def call(self, path: str) -> str:
        """
        Get the size of the file at the given path.

        :param path: Path to the file.
        :return: File size in MB as a string.
        """
        try:
            # Check if the path exists and is a file
            if not os.path.exists(path):
                error(f"The path does not exist or is not a file: {path}")
                raise FileNotFoundError(f"Path does not exist or is not a file: {path}")

            # Get file size in MB
            file_size = (os.path.getsize(path) / 1024) / 1024  # Size in MB
            info(f"File size: {file_size:.2f} MB")
            return f"File size: {file_size:.2f} MB"

        except FileNotFoundError as fnf_error:
            error(f"FileNotFoundError: {fnf_error}")
            raise
        except PermissionError as p_error:
            error(f"PermissionError: {p_error}")
            raise
        except OSError as os_error:
            critical(f"OSError: {os_error}")
            raise
        except Exception as e:
            critical(f"An unexpected error occurred: {e}")
            raise



# Example usage:
# size_instance = Size()
# result = size_instance.call(Path('path/to/your/file'))
# print(result)
