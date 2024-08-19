import sys
import gdown
from pathlib import Path
from abc import ABC, abstractmethod

sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src")
from BrestCancer.ingestdata import ingest_critical, ingest_info
from BrestCancer.utils.size import Size


class IDownload(ABC):
    """
    Interface for downloading files.
    """
    @abstractmethod
    def call(self, url: str, output_path: Path) -> None:
        """
        Abstract method to download a file.

        :param url: URL of the file to be downloaded.
        :param output_path: The destination where the file should be saved.
        """
        pass


class Download(IDownload):
    """
    Concrete implementation of IDownload interface to download files.
    """
    def call(self, url: str, output_path: str, name_dataset: str = "brestcancerset") -> None:
        """
        Downloads a file from the given URL to the specified output path.

        :param url: URL of the file to be downloaded.
        :param output_path: The destination where the file should be saved.
        :param name_dataset: Name for the downloaded file.
        """
        try:
            # Convert output_path to a Path object if it's not already one
            output_path = Path(output_path) # type: ignore
            
            # Extract the file ID from the Google Drive URL
            file_id = url.split('/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'

            # Prepare the output file path
            file_path = output_path / f"{name_dataset}.zip" # type: ignore

            # Download the file using gdown
            ingest_info(f"Starting download from {download_url}")
            gdown.download(download_url, str(file_path), quiet=False)

            # Check the size of the downloaded file
            file_size = Size().call(file_path)
            ingest_info(f"Download completed. File size: {file_size}")

        except Exception as e:
            ingest_critical(f"An error occurred during the download: {e}")
            raise
