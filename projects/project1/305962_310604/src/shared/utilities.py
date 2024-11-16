from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_and_split(dataset):
    X = dataset.data
    y = dataset.target

    return X, y


def create_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_features = make_column_selector(dtype_include="number")
    categorical_features = make_column_selector(dtype_include="object")

    return ColumnTransformer(
        transformers=[
            ("num_pipeline", numeric_pipeline, numeric_features),
            ("cat_pipeline", categorical_pipeline, categorical_features),
        ]
    )
