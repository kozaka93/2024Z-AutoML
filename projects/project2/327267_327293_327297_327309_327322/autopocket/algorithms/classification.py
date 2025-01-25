import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from autopocket.algorithms.base import BaseSearcher, EstimatorWrapper
from scipy.stats import mode


class Classifier(BaseSearcher):
    """
    This class extends BaseSearcher and provides functionality for training and evaluating
    different classification models including Random Forest, Logistic Regression, 
    Decision Tree and Ridge Classifier.

    Parameters
    ----------
    additional_estimators : list, optional (default=None)
        Additional estimator objects to be included in the model selection process.
        Each estimator should implement fit() and predict() methods.

    Attributes
    ----------
    metric_ : str
        The evaluation metric used for model selection (set to 'roc_auc')
    estimators_ : list
        List of classifier wrapper objects including:
        - RandomForestWrapper
        - LogisticRegressionWrapper 
        - DecisionTreeWrapper
        - RidgeClassifierWrapper
    baseline_estimator : sklearn.dummy.DummyClassifier
        A baseline model that predicts the most frequent class
    baseline_strategy : str
        Strategy used by baseline estimator (set to 'most_frequent')

    Methods
    -------
    get_metric()
        Returns the evaluation metric being used
    get_estimators()
        Returns the list of estimator objects
    
    measure_importances(X, y) #out of order
        Calculates feature importance scores using RandomForestClassifier
        
    Notes
    -----
    - Uses ROC AUC as the default metric for model evaluation
    - Includes a dummy classifier as baseline for comparison
    - Can be extended with additional estimators through initialization parameter
    - Measure importances is not in use
    - Feature importance measurement includes a random control variable
    """
    def __init__(self, additional_estimators=None):
        gini = GINI()
        super().__init__(
            gini,
            [
                RandomForestWrapper(),
                LogisticRegressionWrapper(),
                DecisionTreeWrapper(),
                RidgeClassifierWrapper()
            ],
            DummyClassifier(strategy='most_frequent'),
            'most_frequent',
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
        forest = RandomForestClassifier()
        forest.fit(X, y)
        importances = forest.feature_importances_
        return pd.Series(importances, index=feature_names)

class GINI:
    def __call__(self, y_true, y_pred):
        return 2*roc_auc_score(y_true, y_pred)-1
    
    @property
    def base(self):
        return "roc_auc"

    def in_base(self, score):
        return 2*score-1
    
    def __str__(self):
        return "GINI"
    
    def __repr__(self):
        return "GINI"

# The following classes are used to wrap the scikit-learn estimators and their hyperparameters
    
class RandomForestWrapper(EstimatorWrapper):
    def __init__(self):
        super().__init__(
            RandomForestClassifier(),
            {
                "n_estimators": randint(100, 501),      
                "min_samples_leaf": randint(1, 251),    
                "max_samples": uniform(0.5, 0.5),        
                "max_features": uniform(1e-6, 1 - 1e-6),
            },
            "RandomForestClassifier",
            #50
            5
        )

class LogisticRegressionWrapper(EstimatorWrapper):
    def __init__(self, big_data=False):
        self.big_data = big_data
        super().__init__(
            Pipeline(
                [
                    ("scaler",StandardScaler()),
                    ("model",LogisticRegression())
                ]
            ),
            None,
            "LogisticRegression",
            #20
            20
        )

    @property
    def param_distributions_(self):
        params = {
                "model__penalty": ["l2"],
                "model__C": [1e-8,1e-6,1e-4,1e-3,0.01,0.1, 1,10,100,500,1000,5000,10000],
                "model__solver": ['saga'] if self.big_data else ['lbfgs'],
                "model__fit_intercept": [True, False],
                "model__class_weight": ["balanced", None],
                "model__l1_ratio": [None],
                "model__max_iter": [800],
            }
        if "saga" in params["model__solver"]:
            params["model__penalty"] = ["elasticnet", "l1", "l2" ,None] 
        elif "lbfgs" in params["model__solver"]:
            params["model__penalty"] = ["l2", None]

        if "elasticnet" in params["model__penalty"]:
            params["model__l1_ratio"] = uniform(0.1,0.9)

        print("Using", params["model__solver"], "solver", end=".")    
        return params


class DecisionTreeWrapper(EstimatorWrapper):
    def __init__(self):
        super().__init__(
            DecisionTreeClassifier(),
            {
                "max_depth": randint(1, 31),
                "min_samples_split": randint(2, 61),
                "criterion": ["gini", "entropy"],
                "min_samples_leaf": randint(1, 61),
            },
            "DecisionTreeClassifier",
            #100
            20
        )

class RidgeClassifierWrapper(EstimatorWrapper):
    def __init__(self):
        super().__init__(
            RidgeClassifier(),
            {
                "alpha": uniform(0, 1),
                "fit_intercept": [True, False],
                "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                "max_iter": [800],
                "class_weight": ["balanced", None]
            },
            "RidgeClassifier",
            #100
            20
        )
