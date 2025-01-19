# numerai_automl/model_trainer.py

import lightgbm as lgb
from typing import Dict, List
import pandas as pd
from numerai_automl.model_trainers.abstract_model_trainer import AbstractModelTrainer
class LGBMModelTrainer(AbstractModelTrainer):
    """LightGBM model trainer implementation.
    
    Args:
        params (Dict): Dictionary of parameters for LGBMRegressor
    """
    def __init__(self, params: Dict):
        self.params = params
        self.model = lgb.LGBMRegressor(**self.params)
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the LightGBM model.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target values
        """
        self.model.fit(X, y)
        self.is_trained = True
        
    def get_model(self):
        """Get the trained model instance.
        
        Returns:
            lgb.LGBMRegressor: Trained LightGBM model
            
        Raises:
            Exception: If model has not been trained
        """
        if not self.is_trained:
            raise Exception("Model is not trained")
        return self.model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            pd.Series: Model predictions
            
        Raises:
            Exception: If model has not been trained
        """
        if not self.is_trained:
            raise Exception("Model is not trained")
        return self.model.predict(X)
