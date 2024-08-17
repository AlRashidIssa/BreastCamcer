from abc import ABC, abstractmethod
from pathlib import Path
import zipfile
import os

from ingestdata import ingest_critical, ingest_debug, ingest_error, ingest_info, ingest_warning
from utils.size import Size


class IUnzip(ABC):
    """
    Interface for unzipping files.
    """
    @abstractmethod
    def call(self, zip_path: Path, extract_to: Path) -> None:
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
    def call(self, zip_path: Path, extract_to: Path) -> None:
        """
        Unzips a file from the given ZIP path to the specified extraction directory.

        :param zip_path: Path to the ZIP file to be extracted.
        :param extract_to: Directory where the contents should be extracted.
        """
        try:
            # Ensure the ZIP file exists
            if not zip_path.exists() or not zip_path.is_file():
                ingest_error(f"The ZIP file does not exist or is not a file: {zip_path}")
                raise FileNotFoundError(f"ZIP file does not exist or is not a file: {zip_path}")

            # Ensure the output directory exists
            if not extract_to.exists():
                ingest_warning(f"Extraction directory does not exist. Creating: {extract_to}")
                extract_to.mkdir(parents=True, exist_ok=True)

            # Extract the ZIP file
            ingest_info(f"Starting extraction of {zip_path} to {extract_to}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            ingest_info(f"Extraction completed successfully to {extract_to}")

        except zipfile.BadZipFile as bad_zip_error:
            ingest_critical(f"BadZipFile error: The file is not a ZIP file or it is corrupted: {zip_path}")
            raise

        except FileNotFoundError as fnf_error:
            ingest_error(f"FileNotFoundError: {fnf_error}")
            raise

        except Exception as e:
            ingest_critical(f"An unexpected error occurred during extraction: {e}")
            raise


# Example usage:
# unzipper = Unzip()
# unzipper.call(Path("/path/to/your/file.zip"), Path("/path/to/extract"))
