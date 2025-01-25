from autopocket.preprocessing.feature_processing.processing_tools.OutlierHandler import OutlierHandler
from autopocket.preprocessing.feature_processing.processing_tools.FeatureEncoder import FeatureEncoder
from autopocket.preprocessing.feature_processing.processing_tools.FeatureSelector import FeatureSelector


class FeatureProcessor():
    """
    Class to transform a DataFrame by encoding features, handling outliers, and selecting relevant features.
    """

    def __init__(self):
        """
        Initialize the FeatureProcessor class.

        This class uses OutlierHandler to manage outliers, FeatureEncoder to encode features,
        and FeatureSelector to select the most relevant features.
        """
        self.outlierHandler = OutlierHandler()
        self.featureEncoder = FeatureEncoder()
        self.featureSelector = FeatureSelector()

    def feature_process(self, X, ml_task):
        """
        Process the input DataFrame X by performing feature encoding, handling outliers,
        and selecting the most relevant features based on the given machine learning task.

        Parameters:
        - X: pandas DataFrame - The input data to be processed.
        - ml_task: string - The type of machine learning task (e.g., "BINARY_CLASSIFICATION", "LINEAR_REGRESSION").

        Returns:
        - pandas DataFrame: The processed DataFrame with encoded features, outliers handled,
          and irrelevant features removed.

        Processing steps:
        1. Encode the features based on the type of machine learning task.
        2. Handle outliers using the Isolation Forest method.
        3. Select relevant features by removing highly correlated features.
        """
        # 1. Encode features based on the machine learning task
        X = self.featureEncoder.feature_encode(X, ml_task)

        # 2. Handle outliers in the data
        X = self.outlierHandler.handle_outliers(X)

        # 3. Select relevant features based on correlation
        X = self.featureSelector.select_features(X)

        return X
