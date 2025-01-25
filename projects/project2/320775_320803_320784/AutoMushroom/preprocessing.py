import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel

def numerical_preprocess(data):
    num_columns = data.select_dtypes(include=['float64', 'int64']).columns

    if len(num_columns) == 0:
        return data

    imputer = SimpleImputer(strategy='mean')
    data[num_columns] = imputer.fit_transform(data[num_columns])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data[num_columns] = scaler.fit_transform(data[num_columns])

    return data

def categorical_preprocess(data):
    cat_columns = data.select_dtypes(include=['object']).columns

    if len(cat_columns) == 0:
        return data

    imputer = SimpleImputer(strategy='most_frequent')
    data[cat_columns] = imputer.fit_transform(data[cat_columns])


    return data


def feature_select(data, target):
    model = RandomForestClassifier(n_estimators=100, random_state=10)
    model.fit(data, target)
    selector = SelectFromModel(model, threshold="mean") 
    data_selected = selector.transform(data)
    selected_features = data.columns[selector.get_support()]
    data_selected = pd.DataFrame(data_selected, columns=selected_features)
    return data_selected, selected_features

def prep(data, target=None, mode='train', features = None):
    data = numerical_preprocess(data)
    data = categorical_preprocess(data)
    data = pd.get_dummies(data, drop_first=True, dtype=int)
    if mode == 'train':
        data, selected_features = feature_select(data, target)
        return data, selected_features
    if mode == 'test':
        data = data[features]
        return data