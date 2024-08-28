from abc import ABC, abstractmethod
from pathlib import Path
import zipfile
import os
import sys

sys.path.append("/BreastCancer")
from src.utils.logging import critical, error, info, warning

class IUnzip(ABC):
    """
    Interface for unzipping files.
    """
    @abstractmethod
    def call(self, zip_path: str, extract_to: str) -> None:
        """
        Abstract method to unzip a file.

        :param zip_path: Path to the ZIP file to be extracted.
        :param extract_to: Directory where the contents should be extracted.
        """
        pass


class Unzip(IUnzip):
    """
    Concrete implementation of IUnzip interface to unzip files.
    """
    def call(self, zip_path: str, extract_to: str) -> None:
        """
        Unzips a file from the given ZIP path to the specified extraction directory.

        :param zip_path: Path to the ZIP file to be extracted.
        :param extract_to: Directory where the contents should be extracted.
        """
        try:
            # Ensure the ZIP file exists
            if not os.path.exists(zip_path):
                error(f"The ZIP file does not exist or is not a file: {zip_path}")
                raise FileNotFoundError(f"ZIP file does not exist or is not a file: {zip_path}")

            # Ensure the output directory exists
            if not os.path.exists(extract_to):
                warning(f"Extraction directory does not exist. Creating: {extract_to}")

            # Extract the ZIP file
            info(f"Starting extraction of {zip_path} to {extract_to}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            info(f"Extraction completed successfully to {extract_to}")

        except zipfile.BadZipFile as bad_zip_error:
            critical(f"BadZipFile error: The file is not a ZIP file or it is corrupted: {zip_path}")
            raise

        except FileNotFoundError as fnf_error:
            error(f"FileNotFoundError: {fnf_error}")
            raise

        except Exception as e:
            critical(f"An unexpected error occurred during extraction: {e}")
            raise


# Example usage:
# unzipper = Unzip()
# unzipper.call(Path("/path/to/your/file.zip"), Path("/path/to/extract"))
