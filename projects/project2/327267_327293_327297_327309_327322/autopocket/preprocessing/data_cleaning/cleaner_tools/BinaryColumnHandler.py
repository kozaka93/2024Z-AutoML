class BinaryColumnHandler:
    """
    Class responsible for handling binary columns, including detection and conversion to binary format.
    """

    def __init__(self):
        """
        Initialize the BinaryColumnHandler class.

        Attributes:
        - true_values: Set of values that will be interpreted as binary 1.
        - false_values: Set of values that will be interpreted as binary 0.
        """
        self.true_values = {'true', 'yes', '1'}
        self.false_values = {'false', 'no', '0', 'n0'}

    def is_binary_column(self, column):
        """
        Check if a column contains only values that can be converted to binary (1 or 0).

        Parameters:
        - column: pandas Series, the column to be checked.

        Returns:
        - bool: True if the column contains only values that can be mapped to binary 1 or 0, False otherwise.

        Notes:
        - The function considers common textual representations of binary values, such as 'true', 'false', 'yes', 'no', '1', and '0'.
        - Missing values (NaN) are ignored during the check.
        """
        # Get unique values from the column before applying the conversion
        unique_values = column.dropna().unique()
        # Define the possible binary values before conversion
        binary_values = self.true_values | self.false_values

        # Check if all unique values in the column can be mapped to binary values
        return all(
            isinstance(val, str) and val.strip().lower() in binary_values
            for val in unique_values
        )

    def convert_yes_no_to_binary(self, value):
        """
        Convert values like 'yes', 'no', 'true', and 'false' to binary 1 and 0 respectively.

        Parameters:
        - value: str, the input value to be converted.

        Returns:
        - int or original value: The converted binary value (1 or 0), or the original value if no conversion is possible.

        Notes:
        - This method trims leading/trailing spaces and converts the input to lowercase before mapping it.
        - If the input is not a string or does not match any known binary representation, it is returned unchanged.
        """
        if isinstance(value, str):
            value = value.strip().lower()  # Remove any spaces and convert to lowercase
            if value in self.true_values:
                return 1
            elif value in self.false_values:
                return 0
        return value
