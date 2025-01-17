class RedundantColumnRemover:
    """
    Class responsible for identifying and removing redundant columns (constant or fully unique).
    """

    def drop_redundant_columns(self, df):
        """
        Remove columns where all values are constant or unique.

        Parameters:
        - df: pandas DataFrame, the input dataframe to be processed.

        Returns:
        - pandas DataFrame: The dataframe with redundant columns removed.

        Steps:
        - The method first identifies columns where all values are constant (i.e., only one unique value).
        - It then identifies columns where all values are unique (i.e., the number of unique values equals the number of rows).
        - Both types of columns are dropped from the dataframe.

        Notes:
        - Constant columns are those with a single unique value, which offer no additional information.
        - Columns with unique values (equal to the number of rows) are dropped if they are of type 'object'.
        """
        # Identify columns with constant values (only one unique value)
        unique_counts = df.nunique()
        cols_to_drop = unique_counts[unique_counts == 1].index
        df.drop(columns=cols_to_drop, inplace=True)

        # Identify columns with unique values (equal to the number of rows)
        cols_to_drop = unique_counts[(unique_counts == len(df)) & (df.dtypes == 'object')].index
        df.drop(columns=cols_to_drop, inplace=True)

        return df
