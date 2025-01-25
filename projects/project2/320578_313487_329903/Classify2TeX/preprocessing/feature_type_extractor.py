import pandas as pd

class FeatureTypeExtractor:
    '''
    A class which extracts the types of feature (column) in the dataset, in order to understand, which transformations we shiould do.
    This class also has some methods that aim at 
    - encoding (one hot and label), 
    - extracting separating datetime into different features
    - min-max scaling
    '''
    def __init__(self):
        self.CATEGORICAL_ONE_HOT = 'categorical_one_hot'
        self.CATEGORICAL_LABEL = 'categorical_label'
        self.TEXT = 'text'
        self.CONTINIOUS = 'continious'
        self.DISCRETE = 'discrete'
        self.DATETIME = 'datetime'
        self.INDEX = 'index' # If the feature is an index column of the dataset.
        self.UNKNOWN = 'unknown' # If the type of the feature was not recognized by the class.
        self.MAX_CATEGORIES_ONE_HOT = 15 # If the number of categories is less or equals to this number, we will one-hot encode the feature.
        self.MAX_CATEGORIES_LABEL = 50 # If the number of categories is less or equals to this number, we will label encode the feature, otherwise we consider it as a text. 

    def get_feature_type(self, feature_name, dataset):
        '''
        Returns the type of the feature - categorical_one_hot, categorical_label, text, continious, discrete, datetime, index, unknown.
        Args:
            - feature_name - the name of the feature, which type we want to extract.
            - dataset - the dataset, which has the feature.
        Returns:
            - a string, the type of the feature.
        '''

        if feature_name =='index' or feature_name == 'Index' or feature_name == 'ID' or feature_name == 'id':
            return self.INDEX
            
        first_column = dataset.columns[0] 

        x = dataset[feature_name]

        x_type = str(x.dtype)

        if x_type.startswith("float"):
            return self.CONTINIOUS
        
        if x_type.startswith("int") or x_type.startswith("uint"):
            if first_column == x.name and len(x.unique()) == len(x):
                return self.INDEX
            return self.DISCRETE
        
        if x_type.startswith("datetime"):
            return self.DATETIME
        
        if x_type == 'object':
            # try to convert to datetime
            try:
                pd.set_option('mode.chained_assignment', None)
                dataset[feature_name] = pd.to_datetime(dataset[feature_name])
                return self.DATETIME
            except:
                pass
            
            if len(x.unique()) <= self.MAX_CATEGORIES_LABEL:
                if len(x.unique()) <= self.MAX_CATEGORIES_ONE_HOT:
                    return self.CATEGORICAL_ONE_HOT
                else:
                    return self.CATEGORICAL_LABEL
            else:
                if len(x.unique()) == len(x) and first_column == x.name:
                    return self.INDEX
                else:
                    return self.TEXT
        return self.UNKNOWN
    
    
    def one_hot_encode(self, feature_name, dataset):
        '''
        One-hot encodes the categorical feature.
        Args:
            - feature_name - the name of the feature, which should be one-hot encoded.
            - dataset - the dataset, which has the feature.
        Returns:
            - the dataset with the one-hot encoded feature instead of the original feature.
        '''
        if self.get_feature_type(feature_name, dataset) != self.CATEGORICAL_ONE_HOT:
            raise ValueError('One-hot encoding should not ne applied to this feature.')
        
        one_hot = pd.get_dummies(dataset[feature_name], prefix=feature_name)
        
        dataset = dataset.drop(columns=[feature_name])
        dataset = pd.concat([dataset, one_hot], axis=1)
        
        return dataset
    
    
    def label_encode(self, feature_name, dataset):
        '''
        Label encodes the categorical feature.
        Args:
            - feature_name - the name of the feature, which should be label encoded.
            - dataset - the dataset, which has the feature.
        Returns:
            - the dataset with the label encoded feature instead of the original feature.
        '''
        dataset[feature_name] = dataset[feature_name].astype('category')
        dataset[feature_name] = dataset[feature_name].cat.codes
        
        return dataset
    

    def encode_categorical(self, dataset, target_column_name):
        '''
        Encodes the categorical feature.
        Args:
            - dataset - the dataset, to features of which we want to apply the encoding.
            - target_column_name - the name of the target column.
        Returns:
            - the dataset with the encoded categorical features.
        '''
        for feature in dataset.columns:
            feature_type = self.get_feature_type(feature, dataset)
            if feature_type == self.CATEGORICAL_ONE_HOT and feature != target_column_name:
                dataset = self.one_hot_encode(feature, dataset)
            elif feature_type == self.CATEGORICAL_LABEL and feature != target_column_name:
                dataset = self.label_encode(feature, dataset)
        
        return dataset
    
    
    def min_max_scale(self, dataset, target_column_name):
        '''
        Min-max scaling of the features.
        Args:
            - dataset - the dataset, to features of which we want to apply the min-max scaling.
            - target_column_name - the name of the target column.
        Returns:
            - the dataset with the min-max scaled continious features.
        '''
        for feature in dataset.columns:
            feature_type = self.get_feature_type(feature, dataset)
            if (feature_type == self.CONTINIOUS or feature_type == self.DISCRETE or feature_type == self.CATEGORICAL_LABEL) and feature != target_column_name:
                dataset[feature] = (dataset[feature] - dataset[feature].min()) / (dataset[feature].max() - dataset[feature].min())
                
        return dataset
    
    
    def encode_target(self, dataset, target_column_name):
        '''
        Encodes the target column.
        Args:
            - target_column_name - the name of the target column.
        Returns:
            - the encoded target column.
        '''

        replace_dict = {}

        for i, target in enumerate(dataset[target_column_name].unique()):
            replace_dict[target] = i

        dataset[target_column_name] = dataset[target_column_name].replace(replace_dict)
        dataset[target_column_name] = dataset[target_column_name].astype(int)

        print('Target column was encoded as follows:')
        print(replace_dict)

        return dataset
    

    def separate_datetime(self, dataset):
        '''
        Separates the datetime feature into the year, month, day, hour, minute, second features.
        Args:
            - dataset - the dataset, which has the datetime feature.
        Returns:
            - the dataset with the separated datetime features.
        '''
        for feature in dataset.columns:
            feature_type = self.get_feature_type(feature, dataset)
            if feature_type == self.DATETIME:
                dataset[feature + '_year'] = dataset[feature].dt.year
                dataset[feature + '_month'] = dataset[feature].dt.month
                dataset[feature + '_day'] = dataset[feature].dt.day
                dataset[feature + '_hour'] = dataset[feature].dt.hour
                dataset[feature + '_minute'] = dataset[feature].dt.minute
                dataset[feature + '_second'] = dataset[feature].dt.second
                dataset = dataset.drop(columns=[feature])

        return dataset
    
    def bool_to_int(self, dataset):
        '''
        Converts the boolean features to int.
        Args:
            - dataset - the dataset, which has the boolean features.
        Returns:
            - the dataset with the boolean features converted to int.
        '''
        for feature in dataset.columns:
            if dataset[feature].dtype == 'bool':
                dataset[feature] = dataset[feature].astype(int)
                
        return dataset