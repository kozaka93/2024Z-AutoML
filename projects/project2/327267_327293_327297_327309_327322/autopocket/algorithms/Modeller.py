import warnings

from sklearn.exceptions import ConvergenceWarning
from autopocket.algorithms.classification import Classifier
from autopocket.algorithms.regression import Regressor
from scipy.linalg import LinAlgWarning

class Modeller():
    def __init__(self, additional_estimators=None):
        """
        Porządny init.
        """
        self.additional_estimators = additional_estimators
        pass

    def model(self, X, y, ml_type):
        """
        Creates and trains a machine learning model based on the provided data and type.
        This function initializes and fits either a classification or regression model depending
        on the ml_type parameter. It handles warnings during model training and returns the
        best performing model along with results directory.
        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Features matrix for model training.
        y : array-like of shape (n_samples,)
            Target values. Labels for classification or target values for regression.
        ml_type : str
            Type of machine learning task. Supported values:
            - "BINARY_CLASSIFICATION": for binary classification problems
            - Any other value: for regression problems
        Returns:
        -------
        tuple
            A tuple containing:
            - best_model_: The best performing trained model
            - results_dir: Path to directory containing training results
        Notes:
        -----
        - Uses custom warning handler for LinearAlgebra and Convergence warnings
        - For classification, initializes a Classifier with additional estimators
        - For regression, initializes a Regressor with additional estimators
        - Warnings are caught and handled during model fitting
        """

        if ml_type == "BINARY_CLASSIFICATION":  
            m = Classifier(additional_estimators=self.additional_estimators)
            print("Performing binary classification")
        else:
            m = Regressor(additional_estimators=self.additional_estimators)
            print("Performing regression")

        with warnings.catch_warnings():
            warnings.simplefilter('always', LinAlgWarning)
            warnings.simplefilter('always', ConvergenceWarning)
            warnings.showwarning = custom_warning_handler
            m.fit(X, y)
            
        return m.best_model_, m.results_dir ####

shown_warnings = set()

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    """
        Custom warning handler.
    """
    if category == UserWarning:
        return
    if category in shown_warnings:
        return
    shown_warnings.add(category)
    if category == LinAlgWarning:
        print(message, end=". ")
        return
    if category == ConvergenceWarning:
        print("Some models did not converge", end=". ")
        return
    print(f"{category.__name__}: {message}", end=". ")
    return