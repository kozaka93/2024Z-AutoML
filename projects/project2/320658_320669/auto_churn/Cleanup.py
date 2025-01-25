import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Ustawienia logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    transformer do usuwania wartości odstających za pomocą metody IQR.
    """
    def __init__(self, columns=None, unique_value_threshold=3):
        self.columns = columns
        self.unique_value_threshold = unique_value_threshold

    def fit(self, X, y=None):
        self.iqr_bounds = {}
        for col in self.columns:
            if X[col].nunique() > self.unique_value_threshold:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.iqr_bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X_clean = X.copy()
        for col, (lower, upper) in self.iqr_bounds.items():
            X_clean = X_clean[(X_clean[col] >= lower) & (X_clean[col] <= upper)]
        return X_clean

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer do ekstrakcji cech z kolumn datowych.
    """
    def __init__(self, date_columns=None):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.date_columns:
            # Konwersja do formatu datetime jeśli to konieczne
            if not pd.api.types.is_datetime64_any_dtype(X_transformed[col]):
                X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')

            # Ekstrakcja cech datowych
            X_transformed[col + '_year'] = X_transformed[col].dt.year
            X_transformed[col + '_month'] = X_transformed[col].dt.month
            X_transformed[col + '_day'] = X_transformed[col].dt.day
            X_transformed[col + '_weekday'] = X_transformed[col].dt.weekday
            X_transformed = X_transformed.drop(columns=[col])
        return X_transformed

def preprocessing_pipeline(df, target_column, columns_to_drop=None, ordinal_columns=None, ordinal_mappings=None, date_columns=None):
    """
    Przetwarzanie danych, w tym kodowanie porządkowe, obsługa kolumn datowych.

    Parametry:
        df (pd.DataFrame): Dane wejściowe w formie DataFrame.
        target_column (str): Nazwa kolumny docelowej (target).
        ordinal_columns (list): Lista kolumn porządkowych.
        ordinal_mappings (list): Lista mapowań wartości dla kolumn porządkowych.
        date_columns (list): Lista kolumn z datami.

    Zwraca:
        Pipeline: Pełny pipeline do przetwarzania danych.
        pd.DataFrame: Przetworzony zbiór danych.
    """
    logging.info("Rozpoczęcie tworzenia pipeline'u przetwarzania danych.")
    logging.info(f"Początkowy kształt danych: {df.shape}")
    initial_columns = df.shape[1]-1

    # Usunięcie kolumn, które nie są potrzebne do modelowania
    if columns_to_drop:
        logging.info(f"Usuwane kolumny: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop, errors='ignore')

    # Identyfikacja typów cech
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Usunięcie kolumny docelowej z list cech
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Domyślne wartości dla kolumn porządkowych
    if ordinal_columns is None:
        ordinal_columns = []
    if ordinal_mappings is None:
        ordinal_mappings = []

    # Usunięcie kolumn porządkowych z cech kategorycznych
    for col in ordinal_columns:
        if col in categorical_features:
            categorical_features.remove(col)
    # Rozpoznanie kolumn binarnych numerycznych
    binary_columns = [col for col in numeric_features if df[col].nunique() == 2]
    numeric_features = [col for col in numeric_features if col not in binary_columns]

    logging.info(f"Zidentyfikowane cechy numeryczne: {numeric_features}")
    logging.info(f"Zidentyfikowane cechy kategoryczne: {categorical_features}")
    logging.info(f"Zidentyfikowane cechy porządkowe: {ordinal_columns}")
    logging.info(f"Zidentyfikowane cechy binarne: {binary_columns}")

    # Przetwarzanie danych numerycznych: imputacja braków i skalowanie
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    # Jeśli kolumny binarne, to nie skalujemy
    binary_transformer = 'passthrough'

    # Przetwarzanie danych kategorycznych: imputacja braków i one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    # Kodowanie porządkowe dla zmiennych porządkowych
    ordinal_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(categories=ordinal_mappings))
    ])

    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_columns),
        ('bin', binary_transformer, binary_columns)
    ]

    # Ekstrakcja cech z kolumn datowych
    if date_columns:
        logging.info(f"Zidentyfikowane kolumny datowe: {date_columns}")
        date_transformer = DateFeatureExtractor(date_columns=date_columns)
        transformers.append(('date', date_transformer, date_columns))

    # Połączenie procesorów w ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Usuwanie wartości odstających (opcjonalne dla cech numerycznych)
    outlier_remover = OutlierRemover(columns=numeric_features)

    # Pełny pipeline
    pipeline = Pipeline(steps=[
        ('outlier_removal', outlier_remover),
        ('preprocessor', preprocessor)
    ])

    # Zastosowanie pipeline do danych wejściowych
    X = df.drop(columns=[target_column])
    y = df[target_column]


    logging.info("Tworzenie pipeline'u przetwarzania danych...")
    processed_data = pipeline.fit_transform(X)

    # Przygotowanie nazw kolumn wyjściowych
    processed_columns = numeric_features + \
                        list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].
                             get_feature_names_out(categorical_features)) + \
                        ordinal_columns + binary_columns

    logging.info(f"Liczba kolumn po przetwarzaniu: {processed_data.shape[1]} (bez kolumny docelowej).")
    logging.info(f"Kolumny po przetwarzaniu: {processed_columns}")
    added_columns = len(processed_columns) - len(X.columns)
    removed_columns = len(columns_to_drop) if columns_to_drop else 0
    logging.info(f"Liczba stworzonych kolumn: {added_columns}, liczba usuniętych kolumn: {removed_columns}, różnica w liczbie kolumn: {len(processed_columns) - initial_columns}")

    if processed_data.shape[0] < df.shape[0]:
        logging.warning(f"Liczba wierszy zmniejszona z {df.shape[0]} do {processed_data.shape[0]} "
                        f"(usunięto {df.shape[0] - processed_data.shape[0]} wierszy).")
    else:
        logging.info("Liczba wierszy pozostała bez zmian.")

    final_df = pd.DataFrame(processed_data, columns=processed_columns)
    final_df[target_column] = y.loc[final_df.index].values
    logging.info("Pipeline przetwarzania danych został pomyślnie utworzony.")
    logging.info(f"Kształt przetworzonych danych: {final_df.shape}")

    return pipeline, final_df


