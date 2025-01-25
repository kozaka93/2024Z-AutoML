import pandas as pd
from numerai_automl.utils.utils import get_project_root


class DataSaver:
    """
    A class to save Numerai prediction files.

    Attributes:
        project_root (str): Root path of the project
    """

    def __init__(self):
        self.project_root = get_project_root()

    def save_vanila_predictions_data(self, predictions: pd.DataFrame):
        """
        Save vanilla (raw) model predictions.

        Args:
            predictions (pd.DataFrame): DataFrame containing model predictions
        """
        filepath = f"{self.project_root}/predictions/vanila_predictions.parquet"
        predictions.to_parquet(filepath)

    def save_neutralized_predictions_data(self, predictions: pd.DataFrame):
        """
        Save neutralized model predictions.

        Args:
            predictions (pd.DataFrame): DataFrame containing neutralized predictions
        """
        filepath = f"{self.project_root}/predictions/neutralized_predictions.parquet"
        predictions.to_parquet(filepath)

    def save_ensembled_predictions_data(self, predictions: pd.DataFrame):
        """
        Save ensembled model predictions.

        Args:
            predictions (pd.DataFrame): DataFrame containing ensembled predictions
        """
        filepath = f"{self.project_root}/predictions/ensembled_predictions.parquet"
        predictions.to_parquet(filepath)

