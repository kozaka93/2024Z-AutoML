import re


class NumberFormatFixer:
    """
    Class to fix number formats in columns by replacing commas with dots,
    and ensuring the column is considered numeric if it contains valid numbers.
    """

    def fix_column_format(self, column):
        """
        Fix the format of a column by replacing commas with dots in numeric values.

        Parameters:
        - column: pandas Series, the column to be processed.

        Returns:
        - pandas Series: The column with fixed number format.

        Steps:
        - The method checks the first 3 non-null values of the column to determine if they are valid numbers.
        - If the sampled values are valid numbers, it replaces commas with dots and converts the values to floats.
        - Non-string values are returned unchanged.

        Notes:
        - This function does not guarantee conversion of the entire column to a numeric type.
        - It assumes that if the first 3 sampled values are numeric, the entire column can be processed similarly.
        """
        # Check if three random values are valid numbers
        values_to_check = column.dropna().sample(n=min(3, len(column.dropna())), random_state=34)
        if all(self.is_number(value) for value in values_to_check):
            # Replace commas with dots for all values in the column
            column = column.apply(lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x)
        return column

    def is_number(self, value):
        """
        Check if a value is a valid number (integer or float), allowing commas as decimal separators.

        Parameters:
        - value: any type, the value to be checked.

        Returns:
        - bool: True if the value is a valid number, False otherwise.

        Details:
        - The function uses regular expressions to validate strings as numbers.
        - It allows optional commas or dots as decimal points.
        - Non-string values are checked for being numeric without transformation.

        Regex patterns:
        - `^-?\\d+(\\.|,)\\d+$`: Matches decimal numbers with optional negative sign.
        - `^\\d+$`: Matches whole numbers.
        """
        if isinstance(value, str):
            # Allow for an optional comma or dot for decimal point
            return bool(re.match(r'^-?\d+(\.|,)\d+$', value.strip())) or bool(re.match(r'^\d+$', value.strip()))
        return False
