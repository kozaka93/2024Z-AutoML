from autopocket.preprocessing.data_cleaning.cleaner_tools.DateHandler import DateHandler
from autopocket.preprocessing.data_cleaning.cleaner_tools.StringStripper import StringStripper
from autopocket.preprocessing.data_cleaning.cleaner_tools.StringToLowerConverter import StringToLowerConverter
from autopocket.preprocessing.data_cleaning.cleaner_tools.BinaryColumnHandler import BinaryColumnHandler
from autopocket.preprocessing.data_cleaning.cleaner_tools.RedundantColumnRemover import RedundantColumnRemover
from autopocket.preprocessing.data_cleaning.cleaner_tools.DataImputer import DataImputer
from autopocket.preprocessing.data_cleaning.cleaner_tools.NumberFormatFixer import NumberFormatFixer
from autopocket.preprocessing.data_cleaning.cleaner_tools.PatternRemover import PatternRemover


class DataCleaner:
    """
    Class responsible for cleaning and transforming the data in a DataFrame.
    It includes various cleaning steps like handling missing values, formatting dates, converting string cases,
    stripping spaces, removing redundant columns, converting binary columns and removing patterns.
    """

    def __init__(self):
        """
        Initialize the DataCleaner class, which sets up all the necessary tools for cleaning data.

        Attributes:
        - dateHandler: Instance of DateHandler class for date formatting.
        - stringStripper: Instance of StringStripper class for stripping spaces.
        - stringToLowerConverter: Instance of StringToLowerConverter class for converting strings to lowercase.
        - binaryColumnHandler: Instance of BinaryColumnHandler class for handling binary columns.
        - redundantColumnRemover: Instance of RedundantColumnRemover class for removing redundant columns.
        - dataImputer: Instance of DataImputer class for imputing missing values.
        - numberFormatFixer: Instance of NumberFormatFixer class for fixing number formats (comma/dot).
        - patternRemover: Instance of PatternRemover class for removing common patterns.
        """
        self.dateHandler = DateHandler()
        self.stringStripper = StringStripper()
        self.stringToLowerConverter = StringToLowerConverter()
        self.binaryColumnHandler = BinaryColumnHandler()
        self.redundantColumnRemover = RedundantColumnRemover()
        self.dataImputer = DataImputer()
        self.numberFormatFixer = NumberFormatFixer()
        self.patternRemover = PatternRemover()

    def clean(self, X, num_strategy='mean', cat_strategy='most_frequent', fill_value=None):
        """
        Perform a series of data cleaning operations on the input data `X`.

        Parameters:
        - X: pandas DataFrame, the input data to be cleaned.
        - num_strategy: str, strategy for imputing missing numerical values. Default is 'mean'.
        - cat_strategy: str, strategy for imputing missing categorical values. Default is 'most_frequent'.
        - fill_value: value, optional constant to fill for missing values when strategy is 'constant'.

        Returns:
        - pandas DataFrame: The cleaned DataFrame after applying all the cleaning steps.

        Cleaning steps:
        1. Strip leading and trailing spaces from string columns.
        2. Fix number formats by replacing commas with dots in number columns.
        3. Convert all string columns to lowercase.
        4. Remove redundant columns that are constant or have unique values.
        5. Convert 'yes'/'no' and 'true'/'false' columns to binary (1/0).
        6. Convert date columns to a consistent format (yyyyMMdd).
        7. Impute missing values in the DataFrame based on specified strategies for numerical and categorical columns.
        8. Remove common patterns
        """

        # 1. Remove spaces from the beginning and end of string columns
        X = X.apply(self.stringStripper.strip_strings)

        # 2. Fix different number formats (comma vs dot)
        X = X.apply(lambda x: self.numberFormatFixer.fix_column_format(x) if x.dtype == 'object' else x)

        # 3. Convert all string columns to lowercase
        X = X.apply(self.stringToLowerConverter.to_lowercase)

        # 4. Drop redundant columns
        X = self.redundantColumnRemover.drop_redundant_columns(X)

        # 5. Convert yes/no, true/false to binary (only for appropriate columns)
        for col in X.select_dtypes(include=[object]):
            if self.binaryColumnHandler.is_binary_column(X[col]):
                X[col] = X[col].apply(self.binaryColumnHandler.convert_yes_no_to_binary)

        # 6. Convert date columns to a consistent format (yyyyMMdd)
        for col in X.select_dtypes(include=[object]):
            if self.dateHandler.is_date_column(X[col]):
                X[col] = X[col].apply(self.dateHandler.fix_date_format)

        # 7. Handle missing data using DataImputer with provided strategies
        X = self.dataImputer.impute(X, num_strategy, cat_strategy, fill_value)

        # 8. Remove common patterns
        X = self.patternRemover.remove_pattern(X)

        return X
