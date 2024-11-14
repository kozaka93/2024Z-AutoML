from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class DropMissing(BaseEstimator, TransformerMixin):  # drop columns that have >50% NaN
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.columns_missing = []

    def fit(self, X, y=None):
        self.columns_missing = [
            col for col in X.columns if X[col].isna().mean() > self.threshold
        ]
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.columns_missing)


class DropLowVarianceCategorical(
    BaseEstimator, TransformerMixin
):  # drop categorical columns where more than 95% of rows have the same value
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=["object", "category"]).columns:
            top_cat_percentage = X[col].value_counts(normalize=True).max()
            if top_cat_percentage > self.threshold:
                if col != "target":
                    self.columns_to_drop.append(col)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop)


class DropHighCardinality(
    BaseEstimator, TransformerMixin
):  # drop categorical columns that have >50% unique values
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=["object", "category"]).columns:
            unique_count = X[col].nunique()
            total_count = X[col].count()
            unique_percentage = unique_count / total_count
            if unique_percentage > self.threshold:
                self.columns_to_drop.append(col)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop)


# class CustomLabelEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.encoders = {}

#     def fit(self, X, y=None):
#         for column in X.columns:
#             le = LabelEncoder()
#             le.fit(X[column].fillna('Other'))  # Zastąp brakujące wartości na 'Other'
#             self.encoders[column] = le
#         return self

#     def transform(self, X):
#         X_encoded = X.copy()
#         for column, le in self.encoders.items():
#             # Zastępujemy brakujące wartości 'Other'
#             X_encoded[column] = le.transform(X[column].fillna('Other'))

#             # Sprawdzamy, które wartości są nieznane
#             unknown_mask = ~X[column].isin(le.classes_)
#             if unknown_mask.any():
#                 # Dla nieznanych etykiet przypisz wartość, np. -1
#                 X_encoded[column][unknown_mask] = -1

#         return X_encoded
