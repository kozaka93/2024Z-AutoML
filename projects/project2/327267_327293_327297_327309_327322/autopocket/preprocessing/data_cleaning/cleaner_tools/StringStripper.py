

class StringStripper:
    """
    Class responsible for stripping spaces from the beginning and end of strings in a DataFrame column.
    """

    def __init__(self):
        """
        Initialize the StringStripper class.
        """
        pass

    def strip_strings(self, column):
        """
        Perform string stripping on a DataFrame column.

        Parameters:
        - column: pandas Series, the column to be processed.

        Returns:
        - pandas Series: The processed column.
        """

        column = column.str.strip() if column.dtype == 'object' else column

        return column

