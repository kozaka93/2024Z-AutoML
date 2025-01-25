from .base_model import AutoAudioBaseModel
from sklearn.ensemble import GradientBoostingClassifier
from skopt.space import Integer, Real
import pandas as pd
import numpy as np


class AudioGB(AutoAudioBaseModel):
    def __init__(self, random_state: int):
        self.model = GradientBoostingClassifier(random_state=random_state)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(features)

        predictions = np.array(predictions)
        return predictions
    
    def get_model(self):
        return self.model
    
    def get_param_ranges(self, search_method: str):
        if search_method == "random":
            return {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif search_method == "bayes":
            return {
                "n_estimators": Integer(50, 300),
                "max_depth": Integer(3, 10),
                "learning_rate": Real(0.01, 0.2),
                "subsample": Real(0.6, 1.0),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 4),
            }
    
    def set_params(self, **params):
        self.model.set_params(**params)

    def __str__(self) -> str:
        return "Gradient Boosting"
