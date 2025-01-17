import pandas as pd


class FeatureEncoder:
    """
    Class to encode features for machine learning tasks.
    It supports one-hot encoding for categorical features and label encoding for binary categorical features.
    """

    def __init__(self):
        """
        Initialize the FeatureEncoder class.

        Notes:
        - This class does not require any parameters for initialization.
        """
        pass

    def feature_encode(self, X, ml_task):
        """
        Encode features in the DataFrame X based on the provided machine learning task.

        Parameters:
        - X: pandas DataFrame, the input data to be encoded.
        - ml_task: str, the type of machine learning task. Can be "BINARY_CLASSIFICATION" or other tasks.

        Returns:
        - pandas DataFrame: The DataFrame with encoded features.

        Encoding steps:
        1. One-hot encoding for categorical features:
            - For binary classification, drop the first column to avoid multicollinearity.
            - For other tasks, do not drop the first column.
        2. Label encoding for binary categorical features:
            - Convert binary categorical columns (with exactly two unique values) into numeric labels (0 or 1).

        Notes:
        - One-hot encoding is performed using `pd.get_dummies`.
        - Label encoding is done using `pd.factorize`.
        """
        # One-hot encoding for categorical features
        if ml_task == "BINARY_CLASSIFICATION":
            X = pd.get_dummies(X, drop_first=True)
        else:
            X = pd.get_dummies(X, drop_first=False)

        # Label encoding for binary categorical features
        for col in X.select_dtypes(include=[object]):
            if X[col].nunique() == 2:
                X[col] = pd.factorize(X[col])[0]

        return X
