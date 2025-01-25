import pandas as pd


class PatternRemover:
    """
    Class for removing specific patterns from strings in a DataFrame column.

    The class identifies common patterns such as prefixes, suffixes, or patterns in the middle of the string,
    and removes them accordingly. It also attempts to convert the processed strings into numeric values when possible.
    """

    def __init__(self):
        """
        Initialize the PatternRemover class.
        """
        pass

    def remove_pattern(self, X):
        """
        Remove specific patterns from the strings in a DataFrame.

        This method iterates through each column in the DataFrame, processes the strings, and removes common patterns.
        If a column's pattern removal results in multiple columns (e.g., splitting on a middle pattern),
        the new columns are added to the DataFrame and the original column is dropped.

        Parameters:
        - X: pandas DataFrame - Input data with columns containing strings.

        Returns:
        - X: pandas DataFrame - DataFrame with patterns removed and potentially new columns created.
        """
        for column_name in X.columns:
            result = self.remove_column_pattern(X[column_name])

            if isinstance(result, pd.DataFrame):
                # If the result is a DataFrame, assign each new column to the original DataFrame
                for new_col in result.columns:
                    X[new_col] = self.try_convert_to_numeric(result[new_col])

                X.drop(column_name, axis=1, inplace=True)
            else:
                # If it's a single column, assign the result directly
                X[column_name] = self.try_convert_to_numeric(result)

        return X

    def remove_column_pattern(self, column):
        """
        Remove specific patterns from a single column.

        The method first removes common prefixes and suffixes, then checks for repeated patterns in the middle
        of the strings and splits accordingly if necessary.

        Parameters:
        - column: pandas Series - A single column of strings.

        Returns:
        - column: pandas Series - The column with patterns removed or split.
        """
        if column.dtype != 'object':  # If the column is not of string type, return as is
            return column

        # Remove common prefixes and suffixes
        column = self.remove_common_patterns(column)

        # Split on common middle patterns if any
        column = self.split_on_middle_pattern(column)

        return column

    def remove_common_patterns(self, column):
        """
        Removes common prefix and suffix patterns from the column's strings.

        The method identifies the most common prefix and suffix across all strings in the column and removes them.

        Parameters:
        - column: pandas Series - A single column of strings.

        Returns:
        - column: pandas Series - The column with common patterns removed.
        """
        prefix = self.get_common_prefix(column)
        if prefix:
            column = column.str.lstrip(prefix)

        suffix = self.get_common_suffix(column)
        if suffix:
            column = column.str.rstrip(suffix)

        return column

    def get_common_prefix(self, column):
        """
        Finds the most common prefix (starting characters) shared by all entries in the column.

        The prefix is determined by comparing each string in the column and identifying the common starting characters.

        Parameters:
        - column: pandas Series - A single column of strings.

        Returns:
        - prefix: string - The most common prefix shared by all strings.
        """
        if column.empty:
            return ''

        prefix = str(column.iloc[0])

        for item in column:
            item = str(item)
            temp_prefix = ''
            for i in range(min(len(prefix), len(item))):
                if prefix[i] == item[i]:
                    temp_prefix += prefix[i]
                else:
                    break
            prefix = temp_prefix

            if not prefix:  # If no common prefix found
                break

        return prefix

    def get_common_suffix(self, column):
        """
        Finds the most common suffix (ending characters) shared by all entries in the column.

        Similar to the prefix, but the method compares the end of each string in the column.

        Parameters:
        - column: pandas Series - A single column of strings.

        Returns:
        - suffix: string - The most common suffix shared by all strings.
        """
        if column.empty:
            return ''

        suffix = column.iloc[0]

        for item in column:
            temp_suffix = ''
            for i in range(1, min(len(suffix), len(item)) + 1):
                if suffix[-i] == item[-i]:
                    temp_suffix = suffix[-i] + temp_suffix
                else:
                    break
            suffix = temp_suffix

            if not suffix:  # If no common suffix found
                break

        return suffix

    def split_on_middle_pattern(self, column):
        """
        Finds the most common repeating pattern in the middle of each string and splits by this pattern.

        The pattern must be at least 3 characters long and occur in the middle of each string.

        Parameters:
        - column: pandas Series - A single column of strings.

        Returns:
        - new_column_df: pandas DataFrame - A DataFrame with two new columns split by the common pattern.
        """
        common_pattern = self.find_common_middle_pattern(column)

        if not common_pattern:
            return column

        new_columns = []
        for item in column:
            if common_pattern in item:
                left_part, right_part = item.split(common_pattern, 1)  # Split at the first occurrence
                new_columns.append([left_part, right_part])
            else:
                new_columns.append([item, ''])  # If no match, append the item with an empty string

        # Create a DataFrame with two columns
        new_column_df = pd.DataFrame(new_columns, columns=[column.name + "_left", column.name + "_right"])

        return new_column_df

    def find_common_middle_pattern(self, column):
        """
        Identifies the most common repeating pattern in the middle of each string.

        The pattern must be at least 3 characters long and should repeat in all entries.

        Parameters:
        - column: pandas Series - A single column of strings.

        Returns:
        - pattern: string or None - The most common middle pattern or None if no such pattern exists.
        """
        middle_patterns = []

        for item in column:
            if len(item) >= 3:
                for i in range(1, len(item) - 2):  # Start from 1 to avoid prefix and suffix
                    for j in range(i + 3, len(item) + 1):
                        pattern = item[i:j]
                        if len(pattern) >= 3 and pattern in item[i:j]:
                            middle_patterns.append(pattern)

        if not middle_patterns:
            return None

        pattern_counts = {pattern: middle_patterns.count(pattern) for pattern in set(middle_patterns)}

        most_common_pattern = max(pattern_counts.items(), key=lambda x: (x[1], len(x[0])))[0]

        if middle_patterns.count(most_common_pattern) == len(column):
            return most_common_pattern
        else:
            return None

    def try_convert_to_numeric(self, column):
        """
        Attempts to convert the column to numeric values. If conversion fails, the original column is returned.

        Parameters:
        - column: pandas Series - The column to attempt conversion on.

        Returns:
        - column: pandas Series - The numeric column if conversion succeeded, or the original column if it failed.
        """
        try:
            return pd.to_numeric(column, errors='raise')
        except ValueError:
            return column
