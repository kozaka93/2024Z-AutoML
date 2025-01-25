import re
import pandas as pd

class NumericCommaHandler:
    """
    A class to detect and convert numeric values with commas as decimal separators in a pandas DataFrame.
    """

    def __init__(self):
        """
        Initializes the NumericCommaHandler class.
        """
        self.columns_with_commas = []  # Tracks columns with numeric values containing commas

    @staticmethod
    def is_numeric_with_comma(value):
        """
        Checks if a value looks like a number with a comma as a decimal separator.

        Parameters:
        - value (str): The value to check.

        Returns:
        - bool: True if the value matches the pattern, False otherwise.
        """
        if isinstance(value, str):
            return bool(re.match(r'^\d+,\d+$', value))
        return False

    def detect_columns_with_numeric_commas(self, dataframe):
        """
        Detects columns in a DataFrame that contain numeric values with commas as decimal separators.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to analyze.

        Returns:
        - list: List of column names that contain numeric values with commas.
        """
        self.columns_with_commas = [
            col for col in dataframe.columns
            if any(self.is_numeric_with_comma(val) for val in dataframe[col])
        ]
        return self.columns_with_commas

    def convert_numeric_commas(self, dataframe):
        """
        Converts numeric values with commas as decimal separators to standard floats in detected columns.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - pd.DataFrame: The DataFrame with converted values.
        """
        for col in self.columns_with_commas:
            dataframe[col] = dataframe[col].apply(
                lambda x: float(x.replace(',', '.')) if self.is_numeric_with_comma(x) else x
            )
        return dataframe

    def handle_numeric_format(self, dataframe):
        """
        Detects and converts numeric values with commas in a DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - pd.DataFrame: The processed DataFrame with converted values.
        """
        self.detect_columns_with_numeric_commas(dataframe)
        return self.convert_numeric_commas(dataframe)
