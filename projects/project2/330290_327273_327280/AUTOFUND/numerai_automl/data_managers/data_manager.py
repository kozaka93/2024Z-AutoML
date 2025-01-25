import pandas as pd
from numerai_automl.data_managers.data_loader import DataLoader
from numerai_automl.data_managers.data_downloader import DataDownloader
from numerai_automl.data_managers.data_saver import DataSaver
from numerai_automl.utils.utils import get_project_root



class DataManager:
    """Manages data operations for the Numerai AutoML system.
    
    Handles data loading, downloading, saving and transformations for both
    training and prediction workflows.
    
    Args:
        data_version (str): Version of dataset to use (default: "v5.0")
        feature_set (str): Size of feature set to use (default: "small") can be "small", "medium", "all"
    """
    def __init__(self, data_version: str = "v5.0", feature_set: str = "small"):
        self.project_root = get_project_root()
        self.data_loader = DataLoader(data_version, feature_set)
        self.data_downloader = DataDownloader(data_version)
        self.data_saver = DataSaver()

    def get_features(self):
        """Get the list of features used in the dataset."""
        return self.data_loader.get_features()
    
    def load_live_data(self):
        """Load current live tournament data."""
        return self.data_loader.load_live_data()

    def load_train_data_for_base_models(self):
        """Load training data for initial model training."""
        return self.data_loader.load_train_data()
    
    def load_data_for_creating_predictions_for_base_models(self):
        """Load validation data for generating base model predictions."""
        return self.data_loader.load_validation_data()
    
    def save_vanila_predictions_by_base_models(self, predictions: pd.DataFrame):
        """Save raw predictions from base models before neutralization.
        
        Args:
            predictions (pd.DataFrame): Raw model predictions
        """
        self.data_saver.save_vanila_predictions_data(predictions)

    def load_vanila_predictions_data_by_base_models(self):
        """Load raw predictions from base models."""
        return self.data_loader.load_vanila_predictions_data()

    def save_neutralized_predictions_by_base_models(self, neutralized_predictions: pd.DataFrame):
        """Save neutralized predictions from base models.
        
        Args:
            neutralized_predictions (pd.DataFrame): Neutralized model predictions
        """
        self.data_saver.save_neutralized_predictions_data(neutralized_predictions)

    def load_neutralized_predictions_by_base_models(self):
        """Load neutralized predictions from base models."""
        return self.data_loader.load_neutralized_predictions_data()
    
    def load_ranked_neutralized_predictions_by_base_models(self):
        """Load and rank neutralized predictions within each era.
        
        Returns:
            pd.DataFrame: DataFrame with ranked predictions per era
        """
        all_neutralized_predictions = self.load_neutralized_predictions_by_base_models()
        cols = [col for col in all_neutralized_predictions.columns if "neutralized" in col]
        
        neutralized_predictions = all_neutralized_predictions.copy()
        neutralized_predictions[cols] = neutralized_predictions.groupby("era")[cols].rank(pct=True)

        return neutralized_predictions

    def _get_min_and_max_era(self, train_data: pd.DataFrame):
        """Helper method to get minimum and maximum era numbers.
        
        Args:
            train_data (pd.DataFrame): DataFrame containing era column
            
        Returns:
            tuple: (min_era, max_era) as integers
        """
        min_era = int(train_data["era"].str.replace('era', '').min())
        max_era = int(train_data["era"].str.replace('era', '').max())
        return min_era, max_era
    
    def load_train_data_for_ensembler(self):
        """Load first half of eras for ensemble model training.
        
        Returns:
            pd.DataFrame: Training data for ensemble model
        """
        train_data = self.load_ranked_neutralized_predictions_by_base_models()
        min_era, max_era = self._get_min_and_max_era(train_data)
        
        mid_era = (min_era + max_era) // 2
        train_data = train_data[train_data["era"].str.replace('era', '').astype(int) < mid_era]
        return train_data
    
    def load_validation_data_for_ensembler(self):
        """Load second half of eras for ensemble model validation.
        
        Returns:
            pd.DataFrame: Validation data for ensemble model
        """
        validation_data = self.load_ranked_neutralized_predictions_by_base_models()
        min_era, max_era = self._get_min_and_max_era(validation_data)

        mid_era = (min_era + max_era) // 2
        validation_data = validation_data[validation_data["era"].str.replace('era', '').astype(int) >= mid_era]
       
       
        return validation_data
    
    def load_validation_data_for_meta_model(self):
        """Load second half of eras for ensemble model validation.
        
        Returns:
            pd.DataFrame: Validation data for ensemble model
        """
        validation_data = self.data_loader.load_validation_data()
        min_era, max_era = self._get_min_and_max_era(validation_data)

        mid_era = (min_era + max_era) // 2
        validation_data = validation_data[validation_data["era"].str.replace('era', '').astype(int) >= mid_era]
       
       
        return validation_data


   
