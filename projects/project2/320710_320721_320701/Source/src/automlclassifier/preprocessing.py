import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    Klasa odpowiedzialna za wstępne przetwarzanie danych.
    Zawiera:
      - obsługę braków w danych,
      - skalowanie,
      - enkodowanie zmiennych kategorycznych,
    """
    
    def __init__(self, strategy_num="mean", strategy_cat="most_frequent"):
        """
        :param strategy_num: strategia uzupełniania braków w danych numerycznych (mean, median).
        :param strategy_cat: strategia uzupełniania braków w danych kategorycznych (most_frequent, constant).
        """
        valid_num_strategies = ["mean", "median"]
        valid_cat_strategies = ["most_frequent", "constant"]
        
        assert strategy_num in valid_num_strategies, f"Invalid strategy_num '{strategy_num}'. Must be one of {valid_num_strategies}."
        assert strategy_cat in valid_cat_strategies, f"Invalid strategy_cat '{strategy_cat}'. Must be one of {valid_cat_strategies}."
        
        self.strategy_num = strategy_num
        self.strategy_cat = strategy_cat
        self.pipeline = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Buduje pipeline składający się z:
         - SimpleImputer (num i cat)
         - StandardScaler (num)
         - OneHotEncoder (cat)
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.strategy_num)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.strategy_cat, fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        self.pipeline = preprocessor.fit(X, y)

    def transform(self, X: pd.DataFrame):
        """
        Zwraca przetworzone dane w formie tablicy numpy.
        """
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Łączy fit() i transform().
        """
        return self.fit(X, y).transform(X)
