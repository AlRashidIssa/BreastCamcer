import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer/src/BrestCancer")

from pathlib import Path
from ingestdata.download_data import Download
from ingestdata.unzip_data import Unzip
from ingestdata import ingest_info, ingest_error, ingest_critical


def ingest_data(url: str, zip_path: Path, extract_to: Path) -> None:
    """
    Ingests data by downloading and unzipping a dataset.

    :param url: The URL of the dataset to download.
    :param zip_path: The local file path where the downloaded ZIP file will be saved.
    :param extract_to: The directory where the contents of the ZIP file will be extracted.
    
    :raises Exception: If an error occurs during downloading or unzipping.
    """
    try:
        # Step 1: Download the data
        ingest_info(f"Starting data download from URL: {url}")
        Download().call(url=url, output_path=zip_path)
        ingest_info(f"Download completed. File saved to: {zip_path}")

        # Step 2: Unzip the downloaded file
        ingest_info(f"Starting extraction of ZIP file: {zip_path}")
        Unzip().call(zip_path=zip_path, extract_to=extract_to)
        ingest_info(f"Extraction completed. Files extracted to: {extract_to}")

    except Exception as e:
        ingest_critical(f"An error occurred during the data ingestion process: {e}")
        raise


# Example usage:
if __name__ == "__main__":
    # Define the URL for downloading the dataset
    dataset_url = "https://drive.google.com/file/d/1Ywp1-NctzM7BllvMvMe9eQLRGVZgZqAd/view?usp=drive_link"
    
    # Define the paths for saving and extracting
    zip_file_path = Path("/home/alrashidissa/Desktop/BreastCancer/Dataset/downloads/brestcancerset.zip")
    extract_directory = Path("/home/alrashidissa/Desktop/BreastCancer/Dataset/extract")

    try:
        ingest_data(url=dataset_url, zip_path=zip_file_path, extract_to=extract_directory)
    except Exception as e:
        ingest_error(f"Failed to ingest data: {e}")
