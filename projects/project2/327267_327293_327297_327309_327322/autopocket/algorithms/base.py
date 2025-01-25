from abc import abstractmethod
from time import strftime, gmtime
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer, make_scorer
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import json
import os

from autopocket.algorithms.utils import ResultsReader

class BaseSearcher(BaseEstimator):
    """
    A base abstract class for implementing model selection and hyperparameter optimization.
    This class provides a framework for searching the best model among multiple estimators
    using either GridSearchCV or RandomizedSearchCV. It includes functionality for model
    evaluation, feature importance analysis, and result persistence.
    Parameters
    ----------
    metric : str
        The scoring metric to be used for model evaluation (e.g., 'accuracy', 'f1', 'roc_auc').
    estimators : list
        List of EstimatorWrapper objects containing the models to be evaluated.
    dummy_estimator : sklearn.dummy.DummyClassifier or sklearn.dummy.DummyRegressor, optional
        A dummy estimator for baseline comparison.
    dummy_strategy : str, optional
        The strategy to be used by the dummy estimator (e.g., 'stratified', 'most_frequent').
    additional_estimators : list, optional
        Additional EstimatorWrapper objects to be evaluated after the main estimators.
    Attributes
    ----------
    best_model_ : sklearn estimator
        The best performing model found during the search.
    best_score_ : float
        The score of the best performing model.
    best_params_ : dict
        The parameters of the best performing model.
    metric_ : str
        The scoring metric used for evaluation.
    estimators_ : list
        List of EstimatorWrapper objects.
    n_estimators_ : int
        Number of estimators to evaluate.
    results_ : dict
        Dictionary containing the results for each evaluated model.
    results_dir : str
        Directory path where the results are saved.
    Methods
    -------
    fit(X, y)
        Fits multiple models and finds the best performing one.
    predict(X)
        Makes predictions using the best model found.
    save_results()
        Saves the search results to JSON files.
    read_results()
        Reads previously saved results from JSON files.
    create_model_from_json(wrapper_name)
        Creates a model instance from saved parameters.
    measure_importances(X, y)
        Abstract method for measuring feature importances.
    drop_unimportant_features(X, importances)
        Removes features deemed unimportant based on importance scores.
    get_baseline_prediction(y)
        Abstract method for computing baseline predictions.
    Notes
    -----
    - The class implements abstract methods that must be overridden in child classes.
    - Results are automatically saved to a timestamped directory.
    """
    def __init__(self, metric, estimators, dummy_estimator = None, dummy_strategy = None, additional_estimators=None):
        self.best_model_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.metric_ = metric
        self.estimators_ = estimators
        self.n_estimators_ = len(self.estimators_)
        self.results_ = {}
        now = strftime("%Y%m%d_%H%M%S", gmtime())
        self.results_dir = os.path.join(os.getcwd(), f'results_{now}', f'algorithms_results_{now}')
        self.dummy_estimator = dummy_estimator
        self.dummy_strategy = dummy_strategy
        self.additional_estimators = additional_estimators
    
    def fit(self,X,y):
        """
        Fits the best model using GridSearchCV or RandomizedSearchCV.

        This method performs the following steps:
        1. Validates input data
        2. Fits a dummy estimator (if provided) for baseline comparison
        3. Fits multiple estimators specified in self.estimators_
        4. Optionally fits additional estimators if provided
        5. Saves the results

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. The input samples array where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values. The target array where n_samples is the number of samples.

        Returns
        -------
        self : object
            Returns the instance of the classifier.

        Notes
        -----
        The method keeps track of the best performing model based on the specified metric
        in self.metric_. The best score is stored in self.best_score_.

        The method will print progress information including:
        - Dummy estimator performance (if configured)
        - Number of models being fitted
        - Additional estimators being fitted (if any)

        Raises
        ------
        ValueError
            If the input validation fails (through check_X_y)
        """
        check_X_y(X,y)
        X = X.copy()
        y = y.copy()

        if self.dummy_estimator is not None:
            print("Fitting dummy estimator")
            self.dummy_estimator.fit(X,y)
            scorer = get_scorer(self.metric_) if type(self.metric_) == str else make_scorer(self.metric_)
            print(f"Dummy score (strategy: {self.dummy_strategy}):", scorer(self.dummy_estimator, X, y), self.metric_)

        # Depracted in latest version  
        #print("Measuring importances")
        #importances = self.__class__.measure_importances(X,y)
        #top_3_features = importances.nlargest(3)
        #print("Top 3 features by importance:")
        #print(top_3_features)

        self.best_score_ = -np.inf
        print("Fitting", self.n_estimators_ ,"models")

        self.fit_on(self.estimators_, self.n_estimators_, X,y)

        if self.additional_estimators is not None:
            print(f"Fitting {len(self.additional_estimators)} additional estimator{'s' if len(self.additional_estimators) > 1 else ''}")
            self.fit_on(self.additional_estimators, len(self.additional_estimators), X,y, check_best=False)

        self.save_results()
        return self
    
    def fit_on(self, estimators, n_estimators, X,y, check_best=True):
        """
        Helper method for fitting a list of estimators using grid or randomized search.

        Parameters
        ----------
        estimators : list of EstimatorWrapper
            Collection of estimator objects to be fitted.
        n_estimators : int
            Number of estimators to process.
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        check_best : bool, optional
            Flag to check if the model is the best performing one.
        """
        for i,wrapper in enumerate(estimators):
            print(i+1,"/",n_estimators," | Fitting:", wrapper.name_, end=". ")

            if hasattr(wrapper, "big_data"):
                wrapper.big_data = X.shape[0] > 6000

            scorer = self.metric_ if type(self.metric_) == str else self.metric_.base

            if wrapper.n_iter_ is None:
                rs = GridSearchCV(wrapper.estimator_,
                                    wrapper.param_distributions_,
                                    cv=5,
                                    scoring=scorer
                                    )
            else:
                rs = RandomizedSearchCV(wrapper.estimator_, 
                                        wrapper.param_distributions_,
                                        cv=5,
                                        scoring=scorer,
                                        random_state=420,
                                        n_iter=wrapper.n_iter_
                                        )
            rs.fit(X,y)
            best = rs.best_score_ if type(self.metric_) == str else self.metric_.in_base(rs.best_score_)
            print("Best score:", best , self.metric_)

            self.results_[wrapper.name_] = {
                "estimator": rs.best_estimator_,
                "score": best,
                "params": rs.best_params_
            }

            if check_best and best > self.best_score_:
                self.best_score_ = best
                self.best_model_ = rs.best_estimator_
                self.best_params_ = rs.best_params_    

    @abstractmethod
    def get_baseline_prediction(self, y):
        """
            Abstract method for getting the baseline prediction
            Should be implemented in the child class
        """
        pass

    def predict(self, X):
        """
            Predicts the target variable using the best model
            Parameters:
                X - input features
        """
        check_is_fitted(self)
        return self.best_model_.predict(X)
    
    def save_results(self):
        """
        Saves the optimization results for each algorithm wrapper to JSON files.

        The method creates a directory specified by self.results_dir if it doesn't exist,
        and saves individual JSON files containing scores and parameters for each wrapper.

        The saved JSON structure for each wrapper contains:
            - score: The performance metric value
            - params: The optimized hyperparameters

        Files are saved in format: '{wrapper_name}_results.json'

        Returns:
            None

        Raises:
            OSError: If there are permission issues with creating directory or saving files
        """
        os.makedirs(self.results_dir, exist_ok=True)
        results_dir = self.results_dir

        for wrapper_name, result in self.results_.items():
            result_to_save = {
                "score": result["score"],
                "params": result["params"]
            }
            with open(os.path.join(results_dir, f'{wrapper_name}_results.json'), 'w') as f:
                json.dump(result_to_save, f)
        print(f"Saving results to results/algorithms_results")
    
    def read_results(self):
        """
            Read the results from json files
        """
        reader = ResultsReader(self.results_dir)
        return reader.results
    
    def create_model_from_json(self, wrapper_name: str):
        """
            Create a model from the best parameters found in the search
            Not in use yet, but leaved an option for future improvements
        """
        reader = ResultsReader(self.results_dir)
        return reader.create_model_from_json(wrapper_name)
    
    @staticmethod
    @abstractmethod
    def measure_importances(X,y):
        """
            Abstract method for measuring importances
            Should return a pandas Series with feature importances
            Should add a really_random_variable to the dataset
            Should be implemented in the child class
        """
        pass

    @staticmethod
    def drop_unimportant_features(X, importances: pd.Series):
        really_random_importance = importances.get('really_random_variable', None)
        if really_random_importance is not None:
            columns_to_drop = importances[importances < really_random_importance].index.tolist()
            X.drop(columns=columns_to_drop, inplace=True)
            if len(columns_to_drop) > 5:
                print(f"Dropped columns: {columns_to_drop[:5]} ...")
            else:
                print(f"Dropped columns: {columns_to_drop}")
        important_features = [col for col in X.columns if col not in columns_to_drop]

        print("Saving important features to algorithms_results/important_features.json")

        results_dir = BaseSearcher.results_dir
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'important_features.json'), 'w') as f:
            json.dump(important_features, f)
        return X

class EstimatorWrapper(BaseEstimator):
    """
    A wrapper class for scikit-learn estimators to facilitate hyperparameter tuning and model evaluation.
    This class serves as an abstract base class for creating estimator wrappers that can handle hyperparameter
    distributions and perform model fitting and prediction.
    Attributes:
    -----------
    estimator_ : BaseEstimator
        The scikit-learn estimator instance to be wrapped.
    param_distributions : dict
        Dictionary containing parameter distributions for hyperparameter tuning.
    name_ : str
        Name of the estimator.
    n_iter_ : int
        Number of iterations for hyperparameter tuning.
    Methods:
    --------
    param_distributions_:
        Getter for the parameter distributions.
    fit(X, y):
        Fits the estimator to the provided data.
    predict(X):
        Predicts the target variable for the provided data.
    predict_proba(X, y):
        Predicts the probabilities of the target variable for the provided data.
    """
    def __init__(self, estimator, param_distributions, name, n_iter):
        super().__init__()
        self.estimator_ = estimator
        self.param_distributions = param_distributions
        self.name_ = name
        self.n_iter_ = n_iter

    @property
    def param_distributions_(self):
        """
            Getter for param_distributions
        """
        return self.param_distributions
    
    def fit(self, X,y):
        """
            Fits the estimator
        """
        return self.estimator_.fit(X,y)
    
    def predict(self, X):
        """
            Predicts the target variable
        """
        return self.estimator_.predict(X)
    
    def predict_proba(self,X,y):
        """
            Predicts the probabilities of the target variable
        """
        assert hasattr(self.estimator_, "predict_proba")
        return self.estimator_.predict_proba(X)
    
def create_wrapper(estimator, param_distributions, name, n_iter):
    """
    Create a wrapper for a machine learning estimator with randomized hyperparameter search capabilities.

    Parameters
    ----------
    estimator : estimator object
        A machine learning estimator implementing 'fit' and 'predict' methods.
    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions 
        or lists of parameters to try as values for RandomizedSearchCV.
    name : str
        Identifier string for the estimator wrapper.
    n_iter : int
        Number of parameter settings that are sampled in RandomizedSearchCV.
        
    Returns
    -------
    EstimatorWrapper
        A wrapped estimator object with randomized search functionality.
    
    """
    return EstimatorWrapper(estimator, param_distributions, name, n_iter)

    
