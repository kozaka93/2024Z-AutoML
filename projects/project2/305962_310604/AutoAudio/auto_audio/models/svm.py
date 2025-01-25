from .base_model import AutoAudioBaseModel
from sklearn.svm import SVC
import pandas as pd
import numpy as np


class AudioSVM(AutoAudioBaseModel):
    def __init__(self, random_state: int):
        self.model = SVC(random_state=random_state)

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
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto"],
                "degree": [2, 3, 4, 5],
            }
        elif search_method == "bayes":
            return {
                "C": (0.1, 100),
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto"],
                "degree": (2, 5),
            }
            
    def set_params(self, **params):
        self.model.set_params(**params)

    def __str__(self) -> str:
        return "SVM"
