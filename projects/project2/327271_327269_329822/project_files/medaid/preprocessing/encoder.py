import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Encoder:
    """
    A class to detect and encode categorical columns in a pandas DataFrame.
    """

    def __init__(self, target_column):
        """
        Initialize the Encoder.

        Parameters:
        - target_column (str): The name of the target column to handle encoding for.
        """
        self.encoding_info = {}
        self.target_column = target_column
        self.label_encoder = LabelEncoder()
        self.label_encoding_mapping = None  

    def is_categorical(self, column):
        """
        Check if a single column is categorical based on dtype.

        Parameters:
        - column (pd.Series): The column to check.

        Returns:
        - bool: True if the column is categorical, False otherwise.
        """
        if not isinstance(column, pd.Series):
            raise ValueError("Input must be a pandas Series.")

        return column.dtype.name in ['category', 'object']

    def encode(self, dataframe):
        """
        Apply one-hot encoding to categorical columns in the DataFrame and label encoding for the target column if needed.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - pd.DataFrame: A DataFrame with categorical columns one-hot encoded and target column label-encoded if necessary.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        processed_df = dataframe.copy()

        # Check if the target column needs label encoding
        if self.is_categorical(dataframe[self.target_column]):
            unique_values = dataframe[self.target_column].nunique()
            processed_df[self.target_column] = self.label_encoder.fit_transform(dataframe[self.target_column])
            self.label_encoding_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
            self.encoding_info[self.target_column] = {
                "Type": "Target Categorical",
                "Encoded": True,
                "Encoding Method": "Label Encoding",
                "Unique Values": unique_values,
                "Mapping": self.label_encoding_mapping
            }

        categorical_columns = [
            col for col in dataframe.columns 
            if self.is_categorical(dataframe[col]) and col != self.target_column
        ]

        for col in categorical_columns:
            unique_values = dataframe[col].nunique()
            self.encoding_info[col] = {
                "Type": "Categorical",
                "Encoded": True,
                "Encoding Method": "One-Hot Encoding",
                "Unique Values": unique_values,
            }

        if categorical_columns:
            processed_df = pd.get_dummies(
                processed_df, 
                columns=categorical_columns, 
                drop_first=True, 
                dtype=int
            )

        return processed_df

    def get_encoding_info(self):
        """
        Get details about the encoding process.

        Returns:
        - dict: A dictionary containing information about the encoded columns.
        """
        return self.encoding_info
