import pandas as pd
import numpy as np
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Scaler:
    def __init__(self, target_column):
        self.target_column = target_column
        self.scaling_params = {}

    def detect_distribution(self, column):
        """
        Detect the distribution of the data using the Shapiro-Wilk test.
        - If p-value > 0.05, assume normal distribution.
        - Otherwise, assume skewed distribution.
        """
        col_data = column.dropna()
        stat, p_value = shapiro(col_data)
        return 'normal' if p_value > 0.05 else 'skewed'

    def normalize(self, column):
        scaler = MinMaxScaler()
        reshaped_column = column.values.reshape(-1, 1)
        normalized_column = scaler.fit_transform(reshaped_column)

        self.scaling_params[column.name] = {
            'scaling_method': 'min_max',
            'params': {
                'min': scaler.data_min_[0],
                'max': scaler.data_max_[0]
            }
        }
        return normalized_column.flatten()

    def standardize(self, column):
        scaler = StandardScaler()
        reshaped_column = column.values.reshape(-1, 1)
        standardized_column = scaler.fit_transform(reshaped_column)

        self.scaling_params[column.name] = {
            'scaling_method': 'standardization',
            'params': {
                'mean': scaler.mean_[0],
                'std': scaler.scale_[0]
            }
        }
        return standardized_column.flatten()

    def scale_column(self, column):
        """
        Scales the given column based on its distribution.
        - If the column is normally distributed, standardize it.
        - If the column is skewed, normalize it.
        """
        distribution_type = self.detect_distribution(column)
        if distribution_type == 'normal':
            return self.standardize(column)
        else:
            return self.normalize(column)

    def scale(self, dataframe):
        """
        Iteratively scale each numeric column in the DataFrame.
        
        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to scale.
        
        Returns:
        - pd.DataFrame: A DataFrame with scaled numeric columns.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        scaled_df = dataframe.copy()
        for column_name in dataframe.select_dtypes(include=[np.number]).columns:
            if column_name == self.target_column:
                continue
            scaled_df[column_name] = self.scale_column(dataframe[column_name])
        return scaled_df

    def get_scaling_info(self):
        return self.scaling_params
