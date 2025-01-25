from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd


class ClassBalanceHandler:
    def __init__(self):
        """
        A class that selects and applies a method to balance the data based on the number of observations in each class.
        """

    def fit_resample(self, X, y):
        """
        Automatically selects and applies the best method to balance the dataset.

        Args:
            - X - feature matrix
            - y - target labels
        Returns:
            - resampled X and y
        """
        class_counts = y.value_counts()
        majority_class_count = class_counts.max()
        minority_class_count = class_counts.min()
        ratio = majority_class_count / minority_class_count
        total_samples = len(y)

        if ratio <= 2:
            print("Data is already balanced. No resampling applied.")
            return X, y
        elif total_samples > 500000:
            print("Large dataset detected. Applying undersampling.")
            return self._undersample(X, y)
        elif ratio > 10:
            print("Extreme imbalance detected. Applying SMOTE.")
            return self._smote(X, y)
        else:
            print("Moderate imbalance detected. Applying oversampling.")
            return self._oversample(X, y)

    def _oversample(self, X, y):
        """
        Oversamples the minority class to balance class distribution.
        """
        data = pd.DataFrame(X)
        data['target'] = y

        # Separate majority and minority classes
        majority_class = y.value_counts().idxmax()
        minority_class = y.value_counts().idxmin()

        majority_data = data[data['target'] == majority_class]
        minority_data = data[data['target'] == minority_class]

        # Oversample the minority class
        minority_upsampled = resample(
            minority_data,
            replace=True,
            n_samples=len(majority_data)
        )

        # Combine majority and upsampled minority
        balanced_data = pd.concat([majority_data, minority_upsampled])
        return balanced_data.drop(columns=['target']), balanced_data['target']

    def _undersample(self, X, y):
        """
        Undersamples the majority class to balance class distribution.
        """
        data = pd.DataFrame(X)
        data['target'] = y

        # Separate majority and minority classes
        majority_class = y.value_counts().idxmax()
        minority_class = y.value_counts().idxmin()

        majority_data = data[data['target'] == majority_class]
        minority_data = data[data['target'] == minority_class]

        # Undersample the majority class
        majority_downsampled = resample(
            majority_data,
            replace=False,
            n_samples=len(minority_data)
        )

        # Combine minority and downsampled majority
        balanced_data = pd.concat([minority_data, majority_downsampled])
        return balanced_data.drop(columns=['target']), balanced_data['target']

    def _smote(self, X, y):
        """
        Applies SMOTE algorithm to generate synthetic samples.
        """
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)