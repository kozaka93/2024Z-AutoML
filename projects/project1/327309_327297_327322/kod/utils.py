from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import os

CLASSIFIERS = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    XGBClassifier(),
]

DATA_PATH = "../data"

HISTORY_PATH = "../history"

class ModelSuplier():
    def __init__(self, classifiers: Optional[List[Union[BaseEstimator, ClassifierMixin]]] = None):
        self.classifiers = classifiers if classifiers is not None else CLASSIFIERS

        for clf in self.classifiers:
            try:
                assert is_classifier, f"{type(clf)} is not classifier instance"
            except AssertionError as e:
                raise TypeError(e.args)

        self.transformer = self._create_transformer()

    def _create_transformer(self):
        num_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer()),
            ('scale', MinMaxScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="constant", fill_value="missing")),
            ('one-hot', OneHotEncoder(handle_unknown='ignore'))
        ])


        col_trans = ColumnTransformer([
            ('num_pipeline', num_pipeline, make_column_selector(dtype_include = np.number)),
            ('cat_pipeline', cat_pipeline, make_column_selector(dtype_include = np.object_))
        ])

        return col_trans
    
    def _create_pipe(self, clf = Union[BaseEstimator, ClassifierMixin]):
        return (type(clf), Pipeline([("transformer", self.transformer), ("model", clf)]))
    
    @property
    def pipelines(self):
        return [self._create_pipe(clf) for clf in self.classifiers]
    

class DataLoader():
    def __init__(self, data_path: str = None):
        data_path_checked = type(data_path) is str and os.path.isdir(data_path)
        if not data_path_checked: print("Using default path")
        self.path = data_path if data_path_checked else DATA_PATH

    def load(self):
        data = []
        for file in os.listdir(self.path):
            data.append(pd.read_csv(os.path.join(self.path,file)))
        return data
    
    def transform_to_X_y(self, data: pd.DataFrame):
        """
        Last column of the data should be target.
        """
        return data.iloc[:, :-1], LabelEncoder().fit_transform(data.iloc[:,-1])
    
    @property
    def transformed_data(self):
        return [self.transform_to_X_y(data) for data in self.load()]
    

class DataSaver():
    def __init__(self, data_path: str = None):
        data_path_checked = type(data_path) is str and os.path.isdir(data_path)
        if not data_path_checked: print("Using default history path")
        self.path = data_path if data_path_checked else HISTORY_PATH

    def save(self, datasets: List[pd.DataFrame], prefix:str ,indexes:List[str]):
        assert len(datasets) == len(indexes), "Length of datasets and indexes should be equal"
        for i, df in enumerate(datasets):
            df.to_csv(os.path.join(self.path, f'{prefix}_{indexes[i]}.csv'), index=False)

def get_best_params_overall(df):
    df['params_str'] = df['params'].apply(lambda x: str(x))
    grouped_mean = df.groupby(['params_str'])['mean_test_score'].mean().reset_index()
    grouped_mean.sort_values(by='mean_test_score', ascending=False, inplace=True)
    return grouped_mean.iloc[0, 0], grouped_mean.iloc[0, 1]

def get_best_params_per_dataset(df):
    df['params_str'] = df['params'].apply(lambda x: str(x))
    best_params_per_dataset = df.sort_values(['dataset', 'rank_test_score'], ascending=[True, True]).groupby('dataset').first().reset_index()
    best_params_per_dataset.rename(columns={'params_str': 'best_params', 'mean_test_score': 'best_score'}, inplace=True)
    best_params_per_dataset = best_params_per_dataset[['dataset', 'best_params', 'best_score']]
    default_params, _ = get_best_params_overall(df)
    score_for_default_params = df[df['params_str'] == default_params][['dataset', 'mean_test_score']].rename(columns={'mean_test_score': 'default_score'})
    best_params_per_dataset = best_params_per_dataset.merge(score_for_default_params, on='dataset', how='left')
    best_params_per_dataset['abs_tunability'] = best_params_per_dataset['best_score'] - best_params_per_dataset['default_score']
    best_params_per_dataset['rel_tunability (%)'] = best_params_per_dataset['abs_tunability'] / best_params_per_dataset['default_score'] * 100 
    return best_params_per_dataset

def get_best_params_per_dataset_for_measuring_param_tunability(df, history):
    df['params_str'] = df['params'].apply(lambda x: str(x))
    best_params_per_dataset = df.sort_values(['dataset', 'rank_test_score'], ascending=[True, True]).groupby('dataset').first().reset_index()
    best_params_per_dataset.rename(columns={'params_str': 'best_params', 'mean_test_score': 'best_score'}, inplace=True)
    best_params_per_dataset = best_params_per_dataset[['dataset', 'best_params', 'best_score']]
    default_params, _ = get_best_params_overall(history)
    score_for_default_params = history[history['params_str'] == default_params][['dataset', 'mean_test_score']].rename(columns={'mean_test_score': 'default_score'})
    best_params_per_dataset = best_params_per_dataset.merge(score_for_default_params, on='dataset', how='left')
    condition = best_params_per_dataset['best_score'] < best_params_per_dataset['default_score']
    best_params_per_dataset.loc[condition, 'best_score'] = best_params_per_dataset['default_score']
    best_params_per_dataset.loc[condition, 'best_params'] = default_params
    best_params_per_dataset['abs_tunability'] = (best_params_per_dataset['best_score'] - best_params_per_dataset['default_score'])
    best_params_per_dataset['rel_tunability (%)'] = best_params_per_dataset['abs_tunability'] / best_params_per_dataset['default_score'] * 100 
    return best_params_per_dataset






