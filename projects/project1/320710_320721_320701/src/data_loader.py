import pandas as pd
from preprocessing import encode_labels

def load_wine_data():
    df = pd.read_csv('datasets/WineQT.csv')
    y = df['quality']
    X = df.drop(columns='quality')
    y_encoded = encode_labels(y)
    return X, y_encoded

def load_drug_data():
    df = pd.read_csv('datasets/drug.csv')
    X = df.drop(columns='Drug')
    y = df['Drug']
    y_encoded = encode_labels(y)
    return X, y_encoded

def load_iris_data():
    df = pd.read_csv('datasets/Iris.csv')
    y = df["Species"]
    y_encoded = encode_labels(y)
    X = df.drop(columns=["Species", "Id"])
    return X, y_encoded

def load_titanic_data():
    train = pd.read_csv('datasets/train.csv')
    X_train = train.drop(columns=["Survived", "PassengerId"])
    y_train = train["Survived"]
    return X_train, y_train
