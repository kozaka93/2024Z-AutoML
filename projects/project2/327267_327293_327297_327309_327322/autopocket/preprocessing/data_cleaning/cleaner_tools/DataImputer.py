import numpy as np
from sklearn.impute import SimpleImputer


class DataImputer:
    def __init__(self):
        """
        Initialize the DataImputer class.
        The strategy will be chosen dynamically during imputation.
        """

    def impute(self, X, num_strategy='mean', cat_strategy='most_frequent', fill_value=None):
        """
        Perform imputation on the numerical and categorical columns of X using the specified strategies.

        Parameters:
        - X: pandas DataFrame, the input data with missing values.
        - num_cols: List or pandas Index of numerical column names.
        - cat_cols: List or pandas Index of categorical column names.
        - num_strategy: Strategy to use for imputing numerical columns. Default is 'mean'.
                        Available strategies: 'mean', 'median', 'most_frequent', 'constant'.
        - cat_strategy: Strategy to use for imputing categorical columns. Default is 'most_frequent'.
                        Available strategies: 'most_frequent', 'constant'.

        Available strategies for SimpleImputer:
        - 'mean': Replaces missing values with the mean of the column (for numerical data).
        - 'median': Replaces missing values with the median of the column (for numerical data).
        - 'most_frequent': Replaces missing values with the most frequent value (mode) of the column (for categorical data).
        - 'constant': Replaces missing values with a constant value, which can be specified.

        Returns:
        - X: pandas DataFrame with missing values imputed.
        """
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns

        # Validate the strategies
        num_valid_strategies = ['mean', 'median', 'most_frequent', 'constant']
        cat_valid_strategies = ['most_frequent', 'constant']
        if num_strategy not in num_valid_strategies:
            raise ValueError(f"Invalid strategy for numerical columns. Valid options: {num_valid_strategies}")
        if cat_strategy not in cat_valid_strategies:
            raise ValueError(f"Invalid strategy for categorical columns. Valid options: {cat_valid_strategies}")

        # Impute numerical columns
        if len(num_cols) > 0:
            for col in num_cols:
                # Check if the column has exactly two unique values
                if X[col].nunique() == 2:
                    # Use the most frequent value for imputation
                    most_frequent_value = X[col].mode()[0]
                    X[col] = X[col].fillna(most_frequent_value)
                else:
                    # Use SimpleImputer for other cases (mean, median, etc.)
                    num_imputer = SimpleImputer(strategy=num_strategy, fill_value=fill_value)
                    X[col] = num_imputer.fit_transform(X[[col]])


        # Impute categorical columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        return X