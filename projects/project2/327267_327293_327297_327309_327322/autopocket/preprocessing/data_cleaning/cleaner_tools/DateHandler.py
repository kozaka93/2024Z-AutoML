import re
import pandas as pd


class DateHandler:
    """
    Class responsible for handling and formatting dates.
    """
    def __init__(self):
        """
        Initialize the DateHandler class.

        Attributes:
        - date_patterns: A list of tuples, where each tuple contains:
            - A regex pattern for detecting a specific date format in strings.
            - The corresponding date format string for `pd.to_datetime` parsing.
        """
        self.date_patterns = [
            (r"(\d{2})-(\d{2})-(\d{2})", "%d-%m-%y"),  # 07-05-20
            (r"(\d{2})\.(\d{2})\.(\d{2})", "%d.%m.%y"),  # 28.05.10
            (r"(\d{2})\s([a-z]+)\s(\d{2})", "%d %B %y"),  # 11 june 20
            (r"(\d{2})/(\d{2})/(\d{2})", "%d/%m/%y"),  # 25/06/10
            (r"(\d{1,2})\s([a-z]+)\s(\d{2})", "%d %B %y"),  # 4 june 20
            (r"(\d{1,2})/(\d{2})/(\d{2})", "%d/%m/%y"),  # 2/03/20
            (r"(\d{1,2})\.(\d{2})\.(\d{2})", "%d.%m.%y"),  # 3.08.12
            (r"(\d{1,2})\s([a-z]+)\s(\d{2}|\d{4})", "%d %b %Y"),  # 2 Apr 2010 or 2 Apr 10
            (r"(\d{2})[-./](\d{2})[-./](\d{2}|\d{4})", "%d-%m-%Y"),  # 02-04-2010 or 02/04/2010
            (r"(\d{2})\s([a-z]+)\s(\d{2}|\d{4})", "%d %B %Y"),  # 2 April 2010
            (r"(\d{1,2})\.(\d{1,2})\.(\d{2}|\d{4})", "%d.%m.%Y"),  # 02.04.2010 or 2.04.2010
            (r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", "%d/%m/%Y"),  # 02/04/2010 or 2/04/2010
            (r"(\d{1,2})-(\d{2})-(\d{4})", "%d-%m-%Y")  # 1-04-2011
        ]

    def fix_date_format(self, value):
        """
        Convert date strings in various formats to a standardized format.

        Parameters:
        - value: str, the input date string to be converted.

        Returns:
        - int: The standardized date as an integer in the format YYYYMMDD.
        - Original value: If no date match is found or parsing fails.

        Notes:
        - The method attempts to match the input against predefined date patterns.
        - If a match is found, it parses the date and converts it to the format YYYYMMDD.
        - Non-date values or unrecognized formats are returned unchanged.
        """
        if isinstance(value, str):
            value = value.strip().lower()

            # Try to match the date patterns
            for pattern, date_format in self.date_patterns:
                match = re.match(pattern, value)
                if match:
                    try:
                        # Parse the matched date using the appropriate format
                        parsed_date = pd.to_datetime(match.group(0), format=date_format)
                        return int(parsed_date.strftime('%Y%m%d'))  # Return as YYYYMMDD
                    except Exception:
                        continue
        return value  # Return original if no match was found

    def is_date_column(self, column):
        """
        Check if a pandas Series contains dates.

        Parameters:
        - column: pandas Series, the column to be checked.

        Returns:
        - bool: True if the column is likely to contain dates, False otherwise.

        Notes:
        - The function samples up to 3 non-null values from the column.
        - It verifies whether these values match any of the predefined date patterns.
        - If all sampled values are recognized as dates, the column is classified as a date column.
        """
        try:
            sample_values = column.dropna().sample(n=min(3, len(column.dropna())), random_state=42)

            for value in sample_values:
                if isinstance(value, str):
                    if not any(re.match(pattern, value) for pattern, _ in self.date_patterns):
                        return False
                else:
                    return False

            return True
        except Exception:
            return False
