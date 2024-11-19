from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

num_pipeline = Pipeline(steps=[
    ('scale', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('one-hot', OneHotEncoder(drop='first', handle_unknown='error', sparse_output=False))
])

preprocessing = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, make_column_selector(dtype_include=np.number)),
    ('cat_pipeline', cat_pipeline, make_column_selector(dtype_include=[pd.CategoricalDtype(), pd.StringDtype(), np.object_]))
], remainder='passthrough', n_jobs=-1)