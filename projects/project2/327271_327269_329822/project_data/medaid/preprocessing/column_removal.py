import pandas as pd

class ColumnRemover:
    """
    A class to detect and remove non-categorical text columns in a pandas DataFrame,
    including columns with 'id' in their name, highly correlated columns, and more.
    """
    def __init__(self, target_column, categorical_threshold=0.2, correlation_threshold=0.9):
        """
        Initialize the detector with thresholds.

        Parameters:
        - threshold (float): The percentage difference of unique values to total values above which a column is considered categorical.
        - correlation_threshold (float): Threshold for correlation between columns.
        """
        self.categorical_threshold = categorical_threshold
        self.correlation_threshold = correlation_threshold
        self.removal_info = {}
        self.target_column = target_column

    def remove_id_columns(self, dataframe):
        """
        Remove columns that have 'id' in their name (case insensitive), 
        have as many unique values as there are rows in the DataFrame, 
        and are not of type float.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - dataframe (pd.DataFrame): The DataFrame with 'id' columns removed.
        """
        # Identify columns that contain 'id' in the name (case insensitive), 
        # have as many unique values as there are rows, and are not float type
        id_columns_to_remove = [
            col for col in dataframe.columns 
            if 'id' in col.lower() 
            and dataframe[col].nunique() == len(dataframe)
            and dataframe[col].dtype != 'float64'
        ]
        
        # Process each identified column
        for col in id_columns_to_remove:
            # Ensure not removing the target column if it contains 'id'
            if col == self.target_column:
                continue
            # Record removal information for the column
            self.removal_info[col] = {"Removed": True, "Reason": "Contains 'id', has unique values equal to rows, and is not float"}
        
        # Drop identified columns from the dataframe
        dataframe.drop(columns=id_columns_to_remove, inplace=True)
        
        return dataframe

    def remove_highly_correlated_columns(self, dataframe):
        """
        Identify highly correlated numeric columns and remove one from each pair.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - dataframe (pd.DataFrame): The DataFrame with highly correlated columns removed.
        """

        numeric_columns = dataframe.select_dtypes(include=['number']).columns

        correlation_matrix = dataframe[numeric_columns].corr()

        to_remove = set()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.correlation_threshold:
                    col_to_remove = correlation_matrix.columns[i]
                    if col_to_remove not in to_remove:
                        if col_to_remove == self.target_column:
                            continue
                        to_remove.add(col_to_remove)
                        self.removal_info[col_to_remove] = {"Removed": True, "Reason": "High correlation"}

        dataframe.drop(columns=to_remove, inplace=True)
        return dataframe


    def remove_non_categorical_text_columns(self, dataframe):
        """
        Remove non-categorical text columns based on the threshold.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - dataframe (pd.DataFrame): The DataFrame with non-categorical text columns removed.
        """
        text_columns_to_drop = []

        for column_name in dataframe.columns:
            column = dataframe[column_name]
            if column.dtype == 'object':  # Text column
                unique_values = column.nunique()
                total_values = len(column)
                percentage_difference = (unique_values / total_values) * 100
                if percentage_difference > self.categorical_threshold * 100:
                    if column_name == self.target_column:
                        continue
                    text_columns_to_drop.append(column_name)
                    self.removal_info[column_name] = {"Removed": True, "Reason": "To many unique text values"}
                else:
                    self.removal_info[column_name] = {"Removed": False}

        dataframe.drop(columns=text_columns_to_drop, inplace=True)
        return dataframe

    def remove(self, dataframe):
        """
        Sequentially remove 'id' columns, highly correlated columns, and non-categorical text columns.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - dataframe (pd.DataFrame): The DataFrame after processing.
        """

        for column_name in dataframe.columns:
            self.removal_info[column_name] = {"Removed": False}

        # Step 1: Remove 'id' columns
        dataframe = self.remove_id_columns(dataframe)

        # Step 2: Remove highly correlated columns
        dataframe = self.remove_highly_correlated_columns(dataframe)

        # Step 3: Remove non-categorical text columns
        dataframe = self.remove_non_categorical_text_columns(dataframe)

        return dataframe

    def get_removal_info(self):
        """
        Returns a dictionary with column names and whether they were removed or not.
        """
        return self.removal_info
