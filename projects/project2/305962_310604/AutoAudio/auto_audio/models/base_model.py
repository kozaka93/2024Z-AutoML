import pandas as pd
import numpy as np

class AutoAudioBaseModel:
    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        """Fit the model to the training data."""
        raise NotImplementedError

    def fit_from_audio(self, audios: pd.DataFrame):
        """Fit the model to the training data."""
        raise NotImplementedError

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions on the test data."""
        raise NotImplementedError

    def predict_from_audio(self, audios: pd.DataFrame) -> np.ndarray:
        """Make predictions on the test data."""
        raise NotImplementedError

    def get_data_type(self) -> str:
        return "features"
    
    def get_param_ranges(self, search_method: str):
        """Get the hyperparameter search space."""
        raise NotImplementedError
    
    def get_model(self):
        """Retrieve the underlying model."""
        raise NotImplementedError
    
    def set_params(self, **params):
        """
        Set parameters directly on the underlying model.
        """

    def __str__(self) -> str:
        return "Base Model"
