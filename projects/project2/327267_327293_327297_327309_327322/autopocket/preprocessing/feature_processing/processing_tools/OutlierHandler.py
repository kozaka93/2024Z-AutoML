import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class OutlierHandler:
    """
    Class to detect and handle outliers in a DataFrame.
    This class uses both Isolation Forest and statistical methods to detect outliers and handle them appropriately.
    """

    def __init__(self, method='IQR', contamination=0.1):
        """
        Initialize the OutlierHandler class with the desired method for detecting outliers.

        Parameters:
        - method: str, method to detect outliers. Options: 'IQR' (Interquartile Range) or 'IsolationForest'.
        - contamination: float, proportion of outliers in the dataset (used for IsolationForest).
        """
        self.method = method
        self.contamination = contamination

    def handle_outliers(self, X):
        """
        Handle outliers in the DataFrame X by detecting them and replacing them with appropriate values.

        Parameters:
        - X: pandas DataFrame, the input data containing features that may have outliers.

        Returns:
        - pandas DataFrame: The DataFrame with outliers replaced by the mean (for numerical columns)
          or the most frequent value (for categorical columns).
        """

        if self.method == 'IQR':
            X = self._handle_outliers_iqr(X)
        elif self.method == 'IsolationForest':
            X = self._handle_outliers_isolation_forest(X)
        else:
            raise ValueError(f"Method '{self.method}' is not supported. Use 'IQR' or 'IsolationForest'.")

        return X

    def _handle_outliers_iqr(self, X):
        """
        Handle outliers using the IQR method (Interquartile Range).

        Parameters:
        - X: pandas DataFrame

        Returns:
        - pandas DataFrame: The DataFrame with outliers replaced by the mean (for numerical columns)
        """
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier thresholds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace outliers with the mean
            mean_value = X[col].mean()
            X[col] = np.where((X[col] < lower_bound) | (X[col] > upper_bound), mean_value, X[col])

        # For categorical columns, replace outliers with the most frequent value
        for col in X.select_dtypes(include=[object]).columns:
            most_frequent_value = X[col].mode()[0]
            X[col] = X[col].apply(lambda x: most_frequent_value if pd.isnull(x) or x not in X[col].unique() else x)

        return X

    def _handle_outliers_isolation_forest(self, X):
        """
        Handle outliers using Isolation Forest method.

        Parameters:
        - X: pandas DataFrame

        Returns:
        - pandas DataFrame: The DataFrame with outliers replaced by the mean (for numerical columns)
        """
        # Detect outliers using Isolation Forest
        clf = IsolationForest(contamination=self.contamination)
        clf.fit(X.select_dtypes(include=[np.number]))  # Only fit on numerical columns
        y_outliers = clf.predict(X.select_dtypes(include=[np.number]))

        # Replace outliers with the mean for numerical columns
        for col in X.select_dtypes(include=[np.number]).columns:
            mean_value = X[col].mean()
            X.loc[y_outliers == -1, col] = mean_value

        # Replace outliers with the most frequent value for categorical columns
        for col in X.select_dtypes(include=[object]).columns:
            most_frequent_value = X[col].mode()[0]
            X.loc[y_outliers == -1, col] = most_frequent_value

        return X
