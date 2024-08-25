import unittest
import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")

from unittest.mock import patch, MagicMock
from pathlib import Path
from src.data.load_data import Download


class TestDownload(unittest.TestCase):

    @patch('src.data.load_data.gdown.download')
    @patch('src.data.load_data.Size')
    @patch('src.data.load_data.info')
    @patch('src.data.load_data.critical')
    def test_successful_download(self, mock_critical, mock_info, mock_size, mock_gdown):
        # Arrange
        mock_size_instance = mock_size.return_value
        mock_size_instance.call.return_value = "100 MB"
        mock_gdown.return_value = None  # Simulate successful download

        url = "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=sharing"
        output_path = "/path/to/download"
        name_dataset = "brestcancerset"

        # Convert output_path to a Path object
        output_path = Path(output_path)

        # Create the Download instance
        downloader = Download()

        # Act
        downloader.call(url=url, output_path=str(output_path), name_dataset=name_dataset)

        # Assert
        file_id = "1a2b3c4d5e6f7g8h9i0j"
        download_url = f'https://drive.google.com/uc?id={file_id}'
        file_path = output_path / f"{name_dataset}.zip"

        mock_info.assert_any_call(f"Starting download from {download_url}")
        mock_gdown.assert_called_once_with(download_url, str(file_path), quiet=False)
        mock_size_instance.call.assert_called_once_with(file_path)
        mock_info.assert_any_call("Download completed. File size: 100 MB")
        mock_critical.assert_not_called()

    @patch('src.data.load_data.gdown.download')
    @patch('src.data.load_data.Size')
    @patch('src.data.load_data.info')
    @patch('src.data.load_data.critical')
    def test_download_failure(self, mock_critical, mock_info, mock_size, mock_gdown):
        # Arrange
        mock_gdown.side_effect = Exception("Network error")  # Simulate a failure in download

        url = "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=sharing"
        output_path = "/path/to/download"
        name_dataset = "brestcancerset"

        downloader = Download()

        # Act & Assert
        with self.assertRaises(Exception) as context:
            downloader.call(url=url, output_path=output_path, name_dataset=name_dataset)

        self.assertTrue("Network error" in str(context.exception))

        # Verify that critical logging was called
        mock_critical.assert_called_once_with("An error occurred during the download: Network error")
        mock_info.assert_any_call("Starting download from https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j")
        mock_size.return_value.call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
