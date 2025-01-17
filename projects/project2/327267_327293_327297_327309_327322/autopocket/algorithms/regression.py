import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LassoLars, LassoLarsIC, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from autopocket.algorithms.base import BaseSearcher, EstimatorWrapper
from sklearn.dummy import DummyRegressor

class Regressor(BaseSearcher):
    """
    This class extends BaseSearcher and provides functionality for training and evaluating
    different regression models including Random Forest, Linear Regression, 
    Decision Tree, Lasso, Ridge and Elastic Net.

    Parameters
    ----------
    additional_estimators : list, optional (default=None)
        Additional regression estimators to be included in the model selection process.
        Each estimator should be wrapped in an appropriate EstimatorWrapper class.

    Attributes
    ----------
    metric_ : str
        The evaluation metric used for model selection ('neg_root_mean_squared_error').
    estimators_ : list
        List of regression estimators available for model selection.
    dummy_estimator_ : DummyRegressor
        A simple baseline model that predicts the mean of the target variable.
    dummy_strategy_ : str
        Strategy used by the dummy estimator ('mean').

    Methods
    -------
    fit(X, y)
        Fit the regression model to the training data.
        
    predict(X)
        Make predictions using the best selected model.

    get_metric()
        Return the evaluation metric used for model selection.

    get_estimators()
        Return the list of available estimators.

    measure_importances(X, y)
        Calculate feature importance scores using RandomForestRegressor.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix
        y : array-like
            Target variable
        
        Returns
        -------
        pandas.Series
            Feature importance scores for each variable, including a random control variable

    Notes
    -----
    - Uses negative RMSE as the default metric for model evaluation
    - Includes a dummy classifier as baseline for comparison
    - Can be extended with additional estimators through initialization parameter
    - Measure importances is not in use
    - Feature importance measurement includes a random control variable

    """
    def __init__(self, additional_estimators=None):
        super().__init__(
            "neg_root_mean_squared_error",
            [
                DecisionTreeWrapper(),
                RandomForestWrapper(),
                LinearRegressionWrapper(),
                LassoWrapper(),
                ElasticNetWrapper(),
                RidgeWrapper(),
                LassoLarsICWrapper()
            ],
            DummyRegressor(strategy='mean'),
            'mean',
            additional_estimators
        )
    def get_metric(self):
        return self.metric_

    def get_estimators(self):
        return self.estimators_

    @staticmethod
    def measure_importances(X, y):
        X = X.copy()
        X["really_random_variable"] = np.random.rand(X.shape[0])
        feature_names = X.columns
        forest = RandomForestRegressor()
        forest.fit(X, y)
        importances = forest.feature_importances_
        return pd.Series(importances, index=feature_names)
    
# EstimatorWrapper classes for regression models

class DecisionTreeWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "max_depth": randint(1, 31),
            "min_samples_split": randint(2, 61),
            "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            "min_samples_leaf": randint(1, 61),
        }
        super().__init__(DecisionTreeRegressor(), param_distributions, "DecisionTreeRegressor",5) #100)

class RandomForestWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "n_estimators": randint(100, 501),      
            "min_samples_leaf": randint(1, 251),    
            "max_samples": uniform(0.5, 0.5),        
            "max_features": uniform(1e-6, 1 - 1e-6),
        }
        super().__init__(RandomForestRegressor(), param_distributions, "RandomForestRegressor",5) #50)

class LinearRegressionWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(LinearRegression(), param_distributions, "LinearRegression", None) # None means GridSearchCV

class LassoWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "alpha": uniform(1e-8, 100),
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(Lasso(), param_distributions, "Lasso", 5)#100)

class RidgeWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "alpha": uniform(1e-8, 100),
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(Ridge(), param_distributions, "Ridge", 5) #100)

class LassoLarsWrapper(EstimatorWrapper): #depracted
    def __init__(self):
        param_distributions = {
            "alpha": uniform(1e-8, 100),
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(LassoLars(), param_distributions, "LassoLars", 5) #100)

class LassoLarsICWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "criterion": ['aic', 'bic'],
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(LassoLarsIC(), param_distributions, "LassoLarsIC", None)

class ElasticNetWrapper(EstimatorWrapper):
    def __init__(self):
        param_distributions = {
            "alpha": uniform(1e-8, 100),
            "l1_ratio": uniform(0.1, 0.9),
            "fit_intercept": [True, False],
            "copy_X": [True]
        }
        super().__init__(ElasticNet(), param_distributions, "ElasticNet", 5) #100)
