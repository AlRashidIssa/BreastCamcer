from abc import ABC, abstractmethod
from pathlib import Path
import gdown
import os

from ingestdata import ingest_critical, ingest_debug, ingest_error, ingest_info, ingest_warning
from utils.size import Size


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
    def call(self, url: str, output_path: Path) -> None:
        """
        Downloads a file from the given URL to the specified output path.

        :param url: URL of the file to be downloaded.
        :param output_path: The destination where the file should be saved.
        """
        try:
            # Ensure the output directory exists
            if not output_path.parent.exists():
                ingest_warning(f"Output directory does not exist. Creating: {output_path.parent}")
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Parse file ID from the Google Drive URL
            file_id = url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            download_url = prefix + file_id

            # Download the file using gdown
            ingest_info(f"Starting download from {download_url}")
            gdown.download(download_url, str(output_path), quiet=False)

            # Check the size of the downloaded file
            if output_path.exists() and output_path.is_file():
                file_size = Size().call(output_path)
                ingest_info(f"Download completed. {file_size}")
            else:
                ingest_error(f"File not found after download: {output_path}")
                raise FileNotFoundError(f"File not found after download: {output_path}")

        except Exception as e:
            ingest_critical(f"An error occurred during the download: {e}")
            raise


# Example usage:
# downloader = Download()
# downloader.call("https://drive.google.com/file/d/your_file_id/view?usp=sharing", Path("/path/to/save/your_file"))