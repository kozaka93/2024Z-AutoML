import pandas as pd
import numpy as np

class CorrelationFeaturesHandler:
    def __init__(self, threshold=0.9):
        """
        This class removes highly correlated columns from the DataFrame.

        Args:
            - threshold - correlation threshold above which columns will be removed
        """
        self.threshold = threshold

    def fit_transform(self, dataset):
        """
        Function returns a dataframe with highly correlated columns removed.
        """
        corr_matrix = dataset.corr().abs()  # absolute correlation matrix
        upper_triangle = np.triu(corr_matrix, k=1)  # upper triangle of the correlation matrix
        to_drop = [dataset.columns[i] for i in range(len(dataset.columns))
                   if any(upper_triangle[:, i] > self.threshold)]

        if len(to_drop) > 1:
          print(f'Due to high correlation with other columns, the columns: {to_drop} have been removed.')
        elif len(to_drop) == 1:
          print(f'Due to high correlation with other column, the column: {to_drop} have been removed.')

        return dataset.drop(to_drop, axis = 1)
