import numpy as np

class FeatureSelector:
    """
    Class to select features by analyzing correlations between variables.
    It identifies and removes highly correlated features (above a threshold).
    """

    def __init__(self):
        """
        Initialize the FeatureSelector class.

        Notes:
        - This class does not require any parameters for initialization.
        """
        pass

    def select_features(self, X):
        """
        Select features by removing highly correlated variables.

        Parameters:
        - X: pandas DataFrame, the input data containing features.

        Returns:
        - pandas DataFrame: The DataFrame with selected features (removing highly correlated ones).

        Feature selection steps:
        1. Compute the correlation matrix of the features.
        2. Identify pairs of features with a correlation greater than 0.95.
        3. Drop one feature from each highly correlated pair.

        Notes:
        - The correlation threshold is set to 0.95.
        - `np.triu` is used to get the upper triangle of the correlation matrix, avoiding duplicate checks.
        """
        # Compute the correlation matrix
        corr_matrix = X.corr().abs()

        # Get the upper triangle of the correlation matrix to avoid redundant checks
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

        # Identify columns with high correlation (above 0.85)
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

        # Drop the highly correlated columns
        X = X.drop(columns=to_drop)

        return X
