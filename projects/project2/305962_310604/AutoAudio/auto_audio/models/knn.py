from .base_model import AutoAudioBaseModel
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


class AudioKNN(AutoAudioBaseModel):
    def __init__(self, n_neighbors: int):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(features)
        return predictions
    
    def get_model(self):
        return self.model
    
    def get_param_ranges(self, search_method: str):
        if search_method == "random":
            return {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            }
        elif search_method == "bayes":
            return {
                "n_neighbors": (3, 11),
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            }
            
    def set_params(self, **params):

        self.model.set_params(**params)

    def __str__(self) -> str:
        return "KNN"
