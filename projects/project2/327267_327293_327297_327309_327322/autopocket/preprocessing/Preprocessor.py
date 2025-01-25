import pandas as pd

from autopocket.preprocessing.data_cleaning.DataCleaner import DataCleaner
from autopocket.preprocessing.task_analysing.ColumnTypeAnalyzer import ColumnTypeAnalyzer
from autopocket.preprocessing.feature_processing.FeatureProcessor import FeatureProcessor


class Preprocessor():
    """
    Class for preprocessing the input data, including cleaning, analyzing target column type,
    and feature processing.
    """

    def __init__(self):
        """
        Initialize the Preprocessor class.

        This class uses:
        - DataCleaner to clean the input data.
        - ColumnTypeAnalyzer to analyze the target column type.
        - FeatureProcessor to encode features, handle outliers, and select relevant features.
        """
        self.dataCleaner = DataCleaner()
        self.columnTypeAnalyzer = ColumnTypeAnalyzer()
        self.featureProcessor = FeatureProcessor()

    def preprocess(self, path, target, num_strategy='mean', cat_strategy='most_frequent', fill_value=None):
        """
        Perform full preprocessing on the dataset.

        Parameters:
        - path: string - The path to the CSV file to be loaded.
        - target: string - The name of the target column in the dataset.
        - num_strategy: string - The imputation strategy for numerical columns ('mean', 'median', 'most_frequent', 'constant').
        - cat_strategy: string - The imputation strategy for categorical columns ('most_frequent', 'constant').
        - fill_value: Any - The value to use when filling missing values, if 'constant' strategy is used.

        Returns:
        - X: pandas DataFrame - The processed feature set.
        - y: pandas Series - The target variable.
        - ml_task: string - The type of machine learning task ("BINARY_CLASSIFICATION" or "LINEAR_REGRESSION").

        Processing steps:
        1. Load the dataset and split it into features (X) and target (y).
        2. Analyze the type of the target column to determine the appropriate ML task.
        3. Clean the feature set using the DataCleaner class.
        4. Process the features (encode, handle outliers, and select features) using the FeatureProcessor class.
        """
        # Load dataset
        data = pd.read_csv(path, sep=None, engine="python")

        # Split data into features (X) and target (y)
        X = data.drop(columns=[target])
        y = data[target]

        # Analyze the target column to determine the ML task type
        ml_task, y = self.columnTypeAnalyzer.analyze_column_type(y)

        # Clean the feature set (X)
        X = self.dataCleaner.clean(X, num_strategy, cat_strategy, fill_value)

        # Process features (encode, handle outliers, and select features)
        X = self.featureProcessor.feature_process(X, ml_task)

        return X, y, ml_task
