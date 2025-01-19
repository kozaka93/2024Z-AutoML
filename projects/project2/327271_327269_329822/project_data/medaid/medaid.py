import warnings

import pandas as pd
from keras.src.applications.inception_resnet_v2 import preprocess_input
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from medaid.preprocessing.preprocessing import Preprocessing, preprocess_input_data
from sklearn.model_selection import train_test_split
from medaid.training.train import train
from medaid.reporting.plots import makeplots
from medaid.reporting.mainreporter import MainReporter
from medaid.reporting.predictexplain import PredictExplainer
import pickle
import sys
import os
import numpy as np
import time

class MedAId:
    """
    The MedAId class automates preprocessing, hyperparameter tuning, training, and evaluation of machine learning models.
    It supports multiple algorithms, generates metrics, visualizations, and explanations, and enables easy model management.

    Usage:
    create an instance of MedAId and call the train() method to train the models. Then use the predict() method to make predictions.
    for a detailed report, call the report() method. The save() method saves the instance as a pickle file. If you want to explain a prediction, use the predict_explain() method.

    Parameters:
        dataset_path (str): Path to the dataset (CSV or Excel file).
        target_column (str): Name of the target column for prediction.
        models (list, optional): List of model names to train. Defaults to all allowed models.
        metric (str, optional): Metric to optimize during training. Defaults to "f1".
        path (str, optional): Path to save results. Defaults to the current working directory.
        search (str, optional): Type of hyperparameter search ("random" or "grid"). Defaults to "random".
        cv (int, optional): Number of cross-validation folds. Defaults to 3.
        n_iter (int, optional): Number of iterations for random search. Defaults to 30.
        test_size (float, optional): Proportion of data used for testing. Defaults to 0.2.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        param_grids (dict, optional): Dictionary of hyperparameter grids for each model. Defaults to predefined grids.
        imputer_lr_correlation_threshold (float, optional): Threshold for logistic regression imputer. Defaults to 0.8.
        imputer_rf_correlation_threshold (float, optional): Threshold for random forest imputer. Defaults to 0.2.
        categorical_threshold (float, optional): Threshold to distinguish text columns from categorical ones.
            If the ratio of unique values to total values is above this threshold, the column is considered text and removed.
            Default is 0.2.
        removal_correlation_threshold float, optional): Correlation threshold for removing highly correlated columns 
            (excluding target variable). Only one column from each correlated group is kept. Default is 0.9.
        y_labels (dict, optional): Dictionary of labels for target column classes. Defaults to None.


    Attributes:
        allowed_models (list): List of allowed model names.
        allowed_metrics (list): List of allowed metrics.
        path (str): Directory path for saving results.
        models (list): Models to be trained.
        metric (str): Optimization metric.
        best_models (list): List of best trained models.
        best_models_scores (list): Scores of the best models.
        best_metrics (pd.DataFrame): Metrics for all models.
        X, y: Preprocessed features and labels.
        X_train, X_test, y_train, y_test: Split training and testing datasets.
        preprocess (Preprocessing): Preprocessing utility instance.

    Methods:
        read_data(): Loads the dataset from the specified path.
        preprocess(): Applies preprocessing to the dataset.
        split_and_validate_data(test_size, max_attempts): Splits the data, ensuring all classes are present in training.
        train():
            - Performs preprocessing of the dataset.
            - Splits the data into training and testing sets with stratification, ensuring all classes are present in the training data.
            - Trains multiple machine learning models using hyperparameter tuning (random search or grid search).
            - Evaluates models using specified metrics and cross-validation.
            - Saves the results of training (e.g., metrics, best models, scores) to CSV files.
            - Generates visualizations and plots to summarize training and model performance.
        predict(X, model_id=0): Makes predictions using the specified model.
        models_ranking(): Returns a DataFrame ranking all trained models by performance.
        report(): Generates a comprehensive training report.
        save(): Saves the MedAId instance as a pickle file.
        predict_explain(input_data=None, model=None): Generates a detailed explanation for model predictions.
    """

    allowed_models = ["logistic", "tree", "random_forest", "xgboost", "lightgbm"]
    allowed_metrics = [ "accuracy", "f1", "recall", "precision"] #TODO ktore metryki ?
    def __init__(self
                 , dataset_path
                 , target_column
                 , models=None
                 , metric = "f1"
                 , path = None
                 , search  = 'random'
                 , cv = 3
                 , n_iter = 30
                 , test_size = 0.2
                 , n_jobs = -1
                 , param_grids = None
                 , imputer_lr_correlation_threshold=0.8
                 , imputer_rf_correlation_threshold=0.2
                 , categorical_threshold=0.2
                 , removal_correlation_threshold=0.9
                 , y_labels = None
                 ):

        self.dataset_path = dataset_path
        self.target_column = target_column


        if models is not None:
            if type(models) is not list:
                raise ValueError("models must be a list")
            for model in models:
                if model not in self.allowed_models:
                    raise ValueError(f"model {model} is not allowed, must be one of {self.allowed_models}")
            self.models = models
        else:
            self.models = self.allowed_models



        if metric not in self.allowed_metrics:
            raise ValueError(f"metric {metric} is not allowed, must be one of {self.allowed_metrics}")
        self.metric = metric

        self.best_models_scores = None
        self.best_models = None
        self.best_metrics = None


        if path:
            self.path = path + "/medaid1"
        else:
            self.path = os.getcwd() + "/medaid1"


        counter = 1
        original_path = os.getcwd() + "/medaid"
        while os.path.exists(self.path):
            self.path = f"{original_path}{counter}"
            counter += 1

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + "/results", exist_ok=True)
        os.makedirs(self.path + "/results/models", exist_ok=True)



        if search not in ["random", "grid"]:
            raise ValueError("search must be either 'random' or 'grid'")
        self.search = search

        self.imputer_lr_correlation_threshold = imputer_lr_correlation_threshold
        self.imputer_rf_correlation_threshold = imputer_rf_correlation_threshold
        self.categorical_threshold = categorical_threshold
        self.removal_correlation_threshold = removal_correlation_threshold        
        self.preprocessing = Preprocessing(self.target_column, self.path, self.imputer_lr_correlation_threshold,
                                        self.imputer_rf_correlation_threshold, self.categorical_threshold,
                                        self.removal_correlation_threshold)
        self.y_labels = y_labels

        if type(cv) is not int:
            raise ValueError("cv must be an integer")
        self.cv = cv
        if type(n_iter) is not int:
            raise ValueError("n_iter must be an integer")
        self.n_iter = n_iter

        self.test_size = test_size

        self.df_before = self.read_data()
        self.df = self.read_data()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_jobs = n_jobs

        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

        self.removal_info = None
        self.imputation_info = None
        self.encoding_info = None
        self.scaling_info = None

        if param_grids:
            if type(param_grids) is not dict:
                raise ValueError("param_grids must be a dictionary")
            for model in self.models:
                if model not in param_grids.keys():
                    raise ValueError(f"model {model} is not in param_grids")
            self.param_grids = param_grids
        else:
            self.param_grids = {
                "logistic": {
                    'C': list(np.logspace(-3, 3, 20)),
                    'penalty': ['l2'],
                    'solver': [
                        'saga',
                        'lbfgs',
                        'newton-cg'
                    ]
                },
                "tree": {
                    'max_depth': [3, 4, 5, 7, 9, 11, 13, 15],
                    'min_samples_split': [2, 4, 6, 8, 10],
                    'min_samples_leaf': [1, 2, 3, 4, 5]
                },
                "random_forest": {
                    'n_estimators': [50, 100, 200, 300, 400, 500],
                    'max_depth': [None, 3, 5, 8, 11, 15, 20],
                    'min_samples_split': [2, 3, 4, 6, 8, 10],
                    'min_samples_leaf': [1, 2, 3, 5, 7, 10],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False],
                },

                "xgboost": {
                    'verbosity': [0],
                    'n_estimators': [50, 100, 200, 300, 400, 500],
                    'max_depth': [3, 5, 7, 9, 11, 13, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
                    'subsample': [0.5, 0.7, 0.8, 1],
                    'colsample_bytree': [0.5, 0.75, 1],
                    'colsample_bylevel': [0.5, 0.75, 1],
                    'reg_alpha': [0, 0.01, 0.1, 0.5, 1, 10],
                    'reg_lambda': [0, 0.01, 0.1, 0.5, 1, 10],
                    'gamma': [0, 0.01, 0.1, 0.5, 1],
                    'min_child_weight': [1, 3, 5, 7, 10],
                    'scale_pos_weight': [1, 10, 50, 100],
                    'tree_method': ['auto', 'exact', 'approx', 'hist']
                },

                "lightgbm": {
                    'verbosity': [-1],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200, 300],
                    'num_leaves': [6, 8, 12, 16, 24, 32],
                    'boosting_type': ['gbdt', 'dart', 'goss'],
                    'max_bin': [255, 510, 1023],
                    'random_state': [500],
                    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
                    'subsample': [0.7, 0.75, 0.8, 0.9],
                    'reg_alpha': [0, 0.5, 1, 1.2, 2, 5],
                    'reg_lambda': [0, 0.5, 1, 1.2, 1.4, 2, 5],
                }
            }


    def __repr__(self):
        return f"medaid(X, y, models={self.models}, metric={self.metric}, path={self.path}, search={self.search}, cv={self.cv}, n_iter={self.n_iter})"

    def __str__(self):
        str = "medaid object\n"
        str+=f"metric: {self.metric}\n"
        if self.best_models is not None:
            str+="trained\n"
            str+="models; scores: \n"
            for i in range(len(self.best_models)):
                str+=f"\t- {self.best_models[i]}: {self.best_models_scores[i]}\n"
        else:
            str+="not trained\n"

        return str

    def read_data(self):
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"File not found at {self.dataset_path}. Please make sure the file exists.")
        if self.dataset_path.endswith(".csv"):
            return pd.read_csv(self.dataset_path, sep=None, engine='python')
        if self.dataset_path.endswith(".xlsx") or self.dataset_path.endswith(".xls"):
            return pd.read_excel(self.dataset_path)
        else:
            raise ValueError(f"File format not supported. Please provide a CSV or Excel file.")

    def preprocess(self):
        preprocessed_df = self.preprocessing.preprocess(self.df_before)
        self.y_labels = self.preprocessing.get_target_encoding_info()
        self.removal_info, self.imputation_info, self.encoding_info, self.scaling_info = self.preprocessing.get_preprocessing_info()
        return  preprocessed_df


    def split_and_validate_data(self, test_size=0.2, max_attempts=50):
        all_classes = set(self.y)  

        for attempt in range(max_attempts):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=42 + attempt
            )
            
            train_classes = set(y_train)

            if all_classes.issubset(train_classes):
                return X_train, X_test, y_train, y_test

        raise ValueError("Could not ensure all classes are present in the training set after maximum attempts.")


    def train(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        df = self.preprocess()
        self.X = df.drop(columns=[self.target_column])
        self.y = df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_and_validate_data(test_size=self.test_size)

        best_models, best_models_scores, best_metrics= train(self.X_train, self.y_train,self.X_test, self.y_test,
                                                             self.models, self.metric, self.path, self.search,
                                                             self.cv, self.n_iter, self.n_jobs, self.param_grids)
        self.best_models = best_models
        self.best_models_scores = best_models_scores
        self.best_metrics = best_metrics

        print("\nFinishing up...")

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        makeplots(self)

        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr



        message = "\n" +  "="*10 + "  Training complete  " + "="*10 + "\n"
        for char in message:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.01)



    def predict(self, X, model_id=0):
        warnings.filterwarnings("ignore", category=FutureWarning)
        if type(X) is not pd.DataFrame:
            raise ValueError("X must be a pandas DataFrame")
        X = preprocess_input_data(self, X)
        if model_id is None:
            if self.best_models is None:
                raise ValueError("You need to train the model first")
            model = self.best_models[0]
        else:
            model = self.best_models[model_id]

        prediction = model.predict(X)
        if self.y_labels:
            labels = {v: k for k, v in self.y_labels.items()}
            prediction = [labels[p] for p in prediction]

        return prediction

    def models_ranking(self):
        return self.best_metrics.reset_index(drop=True)

    def report(self):
        MainReporter(self, self.path).generate_report()

    def save(self):
        with open(f"{self.path}/medaid.pkl", 'wb') as f:
            pickle.dump(self, f)

    def predict_explain(self, input_data = None, model= None):
        warnings.filterwarnings("ignore", category=FutureWarning)
        if not model:
            model = self.best_models[0]
        if not input_data:
            input_data = self.df_before.head(1).drop(columns=[self.target_column])
        pe = PredictExplainer(self, model)
        df = self.df_before.drop(columns=[self.target_column])
        html_report = pe.generate_html_report(df, input_data)
        with open(f"{self.path}/prediction_report.html", 'w') as f:
            f.write(html_report)

