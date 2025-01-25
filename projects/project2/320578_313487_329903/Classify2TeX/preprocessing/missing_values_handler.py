from .feature_type_extractor import FeatureTypeExtractor
import pandas as pd

class MissingValuesHandler:
    '''
    This class handles missing values in the dataset.
    If the feature is on this stage, that means it has less than 90% missing values, so we can handle it, not delete it.

    If the feature is categorical, we fill the missing values with the most frequent value.
    If the feature is discrete, we fill the missing values with the median value.
    If the feature is continious, we fill the missing values with the median value.
    '''
    def __init__(self, dataset):
        self.dataset = dataset.copy()

    def fit_transform(self):
        '''
        Fills the missing values in the dataset.
        '''
        feature_type_extractor = FeatureTypeExtractor()
        for feature in self.dataset.columns:
            feature_type = feature_type_extractor.get_feature_type(feature, self.dataset)

            # If the feature has missing values, fill them.
            if self.dataset[feature].isnull().sum() > 0:
                # Fill the missing values in categorical features with the most frequent value.
                if feature_type == feature_type_extractor.CATEGORICAL_LABEL or feature_type == feature_type_extractor.CATEGORICAL_ONE_HOT:
                    most_frequent_value = self.dataset[feature].mode()[0]
                    self.dataset.loc[self.dataset[feature].isnull(), feature] = most_frequent_value
                    print(f'The missing values in the feature "{feature}" are filled with the most frequent value.')

                # Fill the missing values in discrete and continious features with the median value.
                elif feature_type in [feature_type_extractor.DISCRETE, feature_type_extractor.CONTINIOUS]:
                    median_value = self.dataset[feature].median()
                    self.dataset.loc[self.dataset[feature].isnull(), feature] = median_value
                    print(f'The missing values in the feature "{feature}" are filled with the median value.')

                # If the feature type is not supported, raise an error.
                else:
                    raise ValueError(f'The feature type "{feature_type}" is not supported by the class.')
        
        return self.dataset
