from .feature_type_extractor import FeatureTypeExtractor
from scipy.stats import zscore

class OutliersHandler:
    '''
    A class which handles the outliers in the dataset.(we consider only numerical features here) using the Z score method.)
    '''
    def __init__(self, dataset):
        self.dataset = dataset.copy()
        self.TO_DELETE_ROWS_THRESHOLD = 0.001 # If the fraction of rows with outliers is less than this value, we delete the rows, otherwise we fill the outliers with the median value.
        self.DO_NOTHING_THRESHOLD = 0.005 # If the fraction of rows with outliers is more than this value, we do nothing, we dont consider these values as outliers.
        # if the fraction of rows with outliers is between the two thresholds, we fill the outliers with the median value.

    def fit_transform(self):
        '''
        Handles the outliers in numerical features in the dataset.
        '''
        feature_type_extractor = FeatureTypeExtractor()

        for feature in self.dataset.columns:
            feature_type = feature_type_extractor.get_feature_type(feature, self.dataset)
            if feature_type == feature_type_extractor.DISCRETE or feature_type == feature_type_extractor.CONTINIOUS:

                # apply the Z score method to find the outliers number
                z_scores = zscore(self.dataset[feature])
                abs_z_scores = abs(z_scores)
                outliers_number = (abs_z_scores > 3).sum() 

                if outliers_number > 0:
                    if outliers_number/len(self.dataset) < self.TO_DELETE_ROWS_THRESHOLD:
                        self.dataset = self.dataset[(abs_z_scores < 3)] # Delete the rows with outliers.
                        print(f'{outliers_number} outliers in the feature {feature} were deleted.')

                    elif outliers_number/len(self.dataset) < self.DO_NOTHING_THRESHOLD:
                        self.dataset[feature] = self.dataset[feature].apply(lambda x: self.dataset[feature].median() if abs(zscore([x])[0]) > 3 else x) # Fill the outliers with the median value, if there are many outliers in the feature.
                        print(f'{outliers_number} outliers in the feature {feature} were replaced with the median value.')
        
        return self.dataset


    
    