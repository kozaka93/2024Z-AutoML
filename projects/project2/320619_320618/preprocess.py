import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

class DataPreprocessor:
    def __init__(self, missing_strategy, normalization_method, num_features):
        self.missing_strategy = missing_strategy
        self.normalization_method = normalization_method
        self.num_features = num_features

        self.pipeline = None

    def preprocess_pipeline(self, data, target=None):
        # Identify numeric and categorical columns
        numeric_columns = data.select_dtypes(include=['number']).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        # Imputation
        numeric_imputer = SimpleImputer(strategy=self.missing_strategy)
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        # Scaling
        if self.normalization_method == 'standard':
            scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported normalization method.")

        # Encoding
        encoder = OneHotEncoder(handle_unknown='ignore')

        # Feature selection
        feature_selector = SelectKBest(score_func=f_classif, k=self.num_features) if self.num_features else 'passthrough'

        # Create pipelines for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', numeric_imputer),
            ('scaler', scaler)
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('encoder', encoder)
        ])

        # Combine preprocessors into a column transformer
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ]
        )

        # Apply transformations
        data = pd.DataFrame(self.pipeline.fit_transform(data))

        # Feature selection
        if target is not None and self.num_features:
            feature_selector.fit(data, target)
            data = pd.DataFrame(feature_selector.transform(data))

        return data
