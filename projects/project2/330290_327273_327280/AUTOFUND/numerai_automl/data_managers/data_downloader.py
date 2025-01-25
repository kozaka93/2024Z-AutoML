import pandas as pd
from numerapi import NumerAPI
import os
from numerai_automl.utils.utils import get_project_root


class DataDownloader:
    """
    A class to download Numerai dataset files.

    Attributes:
        napi (NumerAPI): Numerai API client instance
        data_version (str): Version of the dataset to download
        project_root (str): Root path of the project
    """

    def __init__(self, data_version: str = "v5.0"):
        """
        Initialize DataDownloader with specified data version.

        Args:
            data_version (str): Version of the dataset to download
        """
        self.napi = NumerAPI()
        self.data_version = data_version
        self.project_root = get_project_root()

        
    def download_all_data(self):
        """Download all required dataset files (features, train, validation, and live data)."""
        files = ["features.json", "train.parquet", "validation.parquet", "live.parquet"]
        for file in files:
            self._download_file(file)

    def _download_file(self, filename: str):
        """
        Helper method to download a single file.

        Args:
            filename (str): Name of the file to download
        """
        filepath = f"{self.project_root}/{self.data_version}/{filename}"
        if not os.path.exists(filepath):
            print(f"Downloading {self.data_version}/{filename}...")
            self.napi.download_dataset(f"{self.data_version}/{filename}", filepath)

    def download_train_data(self):
        self._download_file("train.parquet")

    def download_validation_data(self):
        self._download_file("validation.parquet")

    def download_live_data(self):
        self._download_file("live.parquet")

    def download_features_data(self):
        self._download_file("features.json")
