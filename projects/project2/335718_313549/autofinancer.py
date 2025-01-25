from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, roc_curve, auc, confusion_matrix, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import skew, loguniform
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import dalex as dx
import time
import pandas as pd
import numpy as np

class AutoFinancer:
    def __init__(self, problem_type='binary_classification', metric=None, random_state=2024, k=10, correlation_threshold=0.9,
                 selection_method='both', method='random_search', cv_folds=5, n_iter=10):
        '''
        Parameters:
            problem_type (str): Type of task ('binary_classification', 'multiclass_classification', 'regression');
            metric (str): Metric used for model evaluation ('roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error');
            random_state (int): Random seed;
            k (int): Number of features selected by SelectKBest;
            correlation_threshold (float): Correlation threshold above which one of the two correlated variables is removed;
            selection_method (str): Feature selection method ('random_forest', 'select_k_best', 'both');
            method (str): Hyperparameter optimization method ('random_search', 'grid_search');
            cv_folds (int): Number of cross-validation folds;
            n_iter (int): Number of iterations in randomized search.
        '''
        self.problem_type = problem_type
        self.random_state = random_state
        self.metric = metric
        self.k = k
        self.correlation_threshold = correlation_threshold
        self.selection_method = selection_method
        self.method = method
        self.cv_folds = cv_folds
        self.n_iter = n_iter

    def train(self, X, y):
        '''
        Main method for automatic model training:
        - Data preprocessing
        - Model selecting and optimization
        - Report generation

        Parameters:
            X (array/DataFrame): Input data;
            y (array/Series): Target variable.
        '''
        # Preprocessing
        X_processed, y_processed = self._preprocess(X, y)

        # Model training and optimization
        self._model_selection_and_optimization(X_processed, y_processed)

        # Report generation
        self._generate_report(X_processed, y_processed)

    def predict(self, X, model_name = None):
        '''
        Prediction.

        Parameters:
            X (array/DataFrame): Input data;
            model_name (str): Name of model, if None, best model on training data is used.

        Returns:
            y_pred (Series): Vector of predicted values.
        '''
        # Processing test data
        X_processed = self._preprocess_test_data(X)

        # Computing prediction based on chosen model
        if model_name is None:
            y_pred = self.best_estimator.predict(X_processed)

        else:
            y_pred = self.optimizers[model_name].predict(X_processed)

        return y_pred

    def predict_proba(self, X, model_name = None):
        '''
        Probability prediction (Only for classification).

        Parameters:
            X (array/DataFrame): Input data;
            model_name (str): Name of model, if None, best model on training data is used.

        Returns:
            y_pred_proba (Series): Vector of predicted probabilities.

        Probability prediction (Only for classification).
        '''

        if self.problem_type not in ['binary_classification', 'multiclass_classification']:
            raise ValueError("This method cannot be used for this type of problem.")

        # Processing test data
        X_processed = self._preprocess_test_data(X)

        # Computing probability prediction of based on chosen model
        if model_name is None:
            y_pred_proba = self.best_estimator.predict_proba(X_processed)

        else:
            y_pred_proba = self.optimizers[model_name].predict_proba(X_processed)

        return y_pred_proba

    def predict_and_report(self, X, y, model_name = None, prediction_threshold = 0.5):
        """
        Class prediction / regression prediction and probability prediction (only for classification). It also creates a report for test dataset.

        Parameters:
            X (array/DataFrame): Test data;
            y (array/Series): Target for test data;
            model_name (str): Name of model, if None, best model on training data is used;
            prediction_threshold (float): Threshold used for confusion matrices (only for binary classification).
        """

        # Processing test data
        X_processed, y_true = self._preprocess_test_and_target_data(X, y)

        if model_name is None:
            model_name = self.best_model

        # Computing prediction of based on chosen model
        y_pred = self.predict(X_processed, model_name = model_name)

        # Computing probability prediction of based on chosen model
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            y_pred_proba = self.predict_proba(X_processed, model_name = model_name)

        else:
            y_pred_proba = None

        if self.problem_type == 'binary_classification':
            self._generate_binary_classification_report(model_name, y_true, y_pred, y_pred_proba, prediction_threshold)

        elif self.problem_type == 'multiclass_classification':
            self._generate_multiclass_classification_report(model_name, y_true, y_pred, y_pred_proba)

        elif self.problem_type == 'regression':
            self._generate_regression_report(model_name, y_true, y_pred)


    def _preprocess(self, X, y):
        """
        Preprocessing of dataset including:
            - Converting target variable to Series,
            - Converting input data to DataFrame,
            - Handling missing values in y,
            - Saving descriptive statistics for numerical variables and numbers of classes for target variable (if classification),
            - Processing target variable,
            - Splitting columns to different types,
            - Imputation of missing values,
            - Standardizing numerical data,
            - Encoding categorical data,
            - Feature selection.

        Parameters:
            X (array/DataFrame): Input data;
            y (array/Series): Target variable.

        Return:
             X_copy (DataFrame): Preprocessed input data;
             y_copy (Series): Preprocessed target variable.
        """

        # Converting target variable
        y_copy = self._to_series(y)

        # Converting input data
        X_copy = self._to_dataframe(X)

        # Handling missing values in y
        X_copy, y_copy = self._remove_missing_y(X_copy, y_copy)

        # Saving descriptive statistics for numerical variables and numbers of classes for target variable (if classification)
        self._summarize(X_copy, y_copy)

        # Processing target variable
        y_copy = self._process_y(y_copy)

        # Splitting columns to different types
        X_copy, binary_cols, numerical_cols, categorical_cols, datetime_cols = self._categorize_columns(X_copy)

        # Imputation of missing values
        X_copy = self._impute_missing_values(X_copy, binary_cols, numerical_cols, categorical_cols)

        # Standardizing numerical data
        X_copy = self._standardize(X_copy, numerical_cols)

        # Encoding categorical data
        X_copy = self._encode(X_copy, categorical_cols)

        # Feature selection
        X_copy = self._feature_selection(X_copy, y_copy)

        return X_copy, y_copy

    def _to_series(self, y):
        """
        Convert target variable to Series.

        Parameters:
            y (array/Series): Target variable.

        Returns:
            y_processed (Series): Converted target variable.
        """
        # Creating data copy
        y_copy = y.copy()

        y_copy = pd.Series(y_copy)
        return y_copy

    def _to_dataframe(self, X):
        """
        Converting input data to dataframe.
        Parameters:
            X (array/DataFrame): Input data.

        Returns:
            X_processed (DataFrame): Converted input data.
        """
        # Creating data copy
        X_copy = X.copy()

        X_copy = pd.DataFrame(X_copy)
        return X_copy

    def _summarize(self, X, y):
        """
        Computing basic descriptive statistics,
        number of classes (if problem_type in ['binary_classification', 'multiclass_classification'].

        Parameters:
            X (DataFrame): Input data;
            y (Series): Target variable.
        """
        self.summary = X.describe()

        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            self.class_counts = y.value_counts()

    def _remove_missing_y(self, X, y):
        """
        Remove rows with missing values in the target variable y.

        Parameters:
            X (DataFrame): Input features.
            y (Series): Target variable.

        Returns:
            X_processed (DataFrame): Input features with rows removed where y has missing values.
            y_processed (Series): Target variable with missing values removed.
        """
        data = pd.concat([X, y], axis=1)
        data_clean = data.dropna(subset=[y.name])
        X = data_clean.drop(columns=[y.name])
        y = data_clean[y.name]
        return X, y

    def _process_y(self, y):
        """
        Processing target variable depending on the problem type.

        Parameters:
            y (Series): Target variable.

        Returns:
            y (Series): Processed target variable.
        """

        # Processing target variable based on the problem type
        if self.problem_type == 'binary_classification':
            self.unique_values = sorted(y.unique())
            if len(self.unique_values) == 2:
                y = y.map({self.unique_values[0]: 0, self.unique_values[1]: 1})
            else:
                raise ValueError("Target variable is not binary")

        elif self.problem_type == 'multiclass_classification':
            self.unique_values = sorted(y.unique())
            le = LabelEncoder()
            y = le.fit_transform(y)

        elif self.problem_type == 'regression':
            try:
                y = y.astype(float)
            except ValueError:
                raise ValueError("Target variable must be numeric for regression!")

        else:
            raise ValueError("Unknown problem type!")

        return y

    def _categorize_columns(self, X):
        """
        Categorizing columns based on data types.

        Parameters:
            X (DataFrame): Input data.

        Returns:
            binary_cols (list): List of binary columns;
            numerical_cols (list): List of numerical columns;
            categorical_cols (list): List of categorical columns;
            datetime_cols (list): List of datetime columns.
        """
        binary_cols = []
        numerical_cols = []
        categorical_cols = []
        datetime_cols = []

        for col in X.columns:
            unique_values = X[col].unique()
            if (X[col].dtype == 'object' or X[col].dtype.name == 'category') and len(unique_values) == 2:
                binary_cols.append(col)
                X[col] = X[col].map({unique_values[0]: 0, unique_values[1]: 1}).astype(int)
            elif X[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            elif np.issubdtype(X[col].dtype, np.datetime64):
                datetime_cols.append(col)
            else:
                try:
                    X[col].astype(float)
                    numerical_cols.append(col)
                except ValueError:
                    categorical_cols.append(col)

        return X, binary_cols, numerical_cols, categorical_cols, datetime_cols

    def _impute_missing_values(self, X, binary_cols, numerical_cols, categorical_cols):
        """
        Imputing missing values in the data.

        Parameters:
            X (DataFrame): Input data;
            binary_cols (list): List of binary columns;
            numerical_cols (list): List of numerical columns;
            categorical_cols (list): List of categorical columns.

        Returns:
            X (DataFrame): Data after imputation.
        """
        symmetrical_cols = []
        skewed_cols = []

        for col in numerical_cols:
            if X[col].isnull().sum() < len(X[col]):  # We skip empty columns
                skewness = skew(X[col].dropna())
                if abs(skewness) < 0.5:
                    symmetrical_cols.append(col)
                else:
                    skewed_cols.append(col)

        # Imputation for numerical columns
        if symmetrical_cols:
            num_imputer_mean = SimpleImputer(strategy='mean')
            X[symmetrical_cols] = num_imputer_mean.fit_transform(X[symmetrical_cols])

        if skewed_cols:
            num_imputer_median = SimpleImputer(strategy='median')
            X[skewed_cols] = num_imputer_median.fit_transform(X[skewed_cols])

        if binary_cols:
            imputer_iterative = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=self.random_state))
            X[binary_cols] = imputer_iterative.fit_transform(X[binary_cols])

        if categorical_cols:
            imputer_mode = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = imputer_mode.fit_transform(X[categorical_cols])

        return X

    def _standardize(self, X, numerical_cols):
        """
        Standardizing numerical columns of the dataset X.

        Params:
            X (DataFrame): Input data;
            numerical_cols (list): List of numerical columns.

        Return:
            X (DataFrame): Dataset with standardized numerical columns.
        """
        if numerical_cols:
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        return X

    def _encode(self, X, categorical_cols):
        """
        Label encoding of categorical columns of the dataset X.

        Params:
            X (DataFrame): Input data;
            categorical_cols (list): List of categorical columns.

        Return:
            X (DataFrame): Dataset with encoded categorical columns.
        """
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        return X

    def _feature_selection(self, X, y):
        """
        Feature selection using correlation and statistical methods.

        Parameters:
            X (DataFrame): Input data;
            y (Series): Target variable.

        Returns:
            X (DataFrame): Data after feature selection.
        """
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.correlation_threshold)]
        X = X.drop(columns=to_drop)

        selected_features = []

        # Feature selection based on the chosen method
        if self.selection_method in ['random_forest', 'both']:
            rf = RandomForestClassifier(random_state=2024)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            rf_selected = importances.nlargest(self.k).index.tolist()
            selected_features.append(set(rf_selected))

        if self.selection_method in ['select_k_best', 'both']:
            skb = SelectKBest(score_func=f_classif, k=self.k)
            skb.fit(X, y)
            skb_selected = X.columns[skb.get_support()].tolist()
            selected_features.append(set(skb_selected))

        # Combining results of both methods (if 'both' is chosen)
        if self.selection_method == 'both':
            self.final_features = list(set.intersection(*selected_features))
        else:
            self.final_features = list(selected_features[0])

        # Returning selected features
        X = X[self.final_features]
        return X

    def _model_selection_and_optimization(self, X, y):
        """
        Model selection and optimization with cross-validation including:
            - Defining models and parameters,
            - Defining scoring metrics,
            - Training and optimizing models.

        Parameters:
            X (DataFrame): Input data;
            y (Series): Target variable.
        """
        # Defining models and parameters
        models, grid_param_grids, random_param_grids = self._define_models_and_params()

        # Defining scoring metrics
        eval_scoring, scoring = self._define_scoring()

        # Training and optimizing models
        self._optimize(X, y, models, grid_param_grids, random_param_grids, eval_scoring, scoring)

    def _define_models_and_params(self):
        """
        Defining models and hyperparameter search spaces.

        Returns:
            models (dict): Dictionary of models;
            grid_param_grids (dict): Dictionary of hyperparameters for Grid Search;
            random_param_grids (dict): Dictionary of hyperparameters for Random Search.
        """
        if self.problem_type == 'binary_classification':
            models = {
                'RandomForest': RandomForestClassifier(random_state=self.random_state),
                'DecisionTree': DecisionTreeClassifier(random_state=self.random_state),
                'XGBoost': XGBClassifier(random_state=self.random_state),
                'GradientBoosting': GradientBoostingClassifier(random_state=self.random_state),
                'LogisticRegression': LogisticRegression(max_iter=500, penalty=None, solver='saga', random_state=self.random_state),
                'LassoRegression': LogisticRegression(max_iter=500, penalty='l1', solver='saga', random_state=self.random_state),
                'RidgeRegression': LogisticRegression(max_iter=500, penalty='l2', solver='saga', random_state=self.random_state),
                'LDA': LinearDiscriminantAnalysis(store_covariance = True),
                'QDA': QuadraticDiscriminantAnalysis(store_covariance = True)
            }

            grid_param_grids = {
                'RandomForest': {'n_estimators': [50, 100, 200],
                                  'max_depth': [None, 10, 20],
                                  'min_samples_split': [2, 5, 10]},
                'DecisionTree': {'max_depth': [None, 10, 20],
                                  'min_samples_split': [2, 5, 10]},
                'XGBoost': {'n_estimators': [50, 100, 200],
                            'max_depth': [3, 6, 10],
                            'learning_rate': [0.01, 0.1, 0.2]},
                'GradientBoosting': {'n_estimators': [50, 100, 200],
                                      'max_depth': [3, 6, 10],
                                      'learning_rate': [0.01, 0.1, 0.2]},
                'LogisticRegression': {},
                'LassoRegression': {'C': [0.1, 1, 10]},
                'RidgeRegression': {'C': [0.1, 1, 10]},
                'LDA': {'solver': ['svd', 'lsqr', 'eigen']},
                'QDA': {'reg_param': [0, 0.001, 0.01]}
            }

            random_param_grids = {
                'RandomForest': {'n_estimators': list(range(1, 501)),
                                  'max_depth': list(range(1, 11)),
                                  'min_samples_split': list(range(2, 11))},
                'DecisionTree': {'max_depth': list(range(1, 11)),
                                  'min_samples_split': list(range(2, 11))},
                'XGBoost': {'n_estimators': list(range(1, 501)),
                            'max_depth': list(range(1, 11)),
                            'learning_rate': loguniform(1e-10, 1)},
                'GradientBoosting': {'n_estimators': list(range(1, 501)),
                                      'max_depth': list(range(1, 11)),
                                      'learning_rate': loguniform(1e-10, 1)},
                'LogisticRegression': {},
                'LassoRegression': {'C': loguniform(1e-10, 1e+10)},
                'RidgeRegression': {'C': loguniform(1e-10, 1e+10)},
                'LDA': {'solver': ['svd', 'lsqr', 'eigen']},
                'QDA': {'reg_param': loguniform(1e-10, 1)}
            }

        elif self.problem_type == 'multiclass_classification':
            models = {
                'RandomForest': OneVsRestClassifier(RandomForestClassifier(random_state=self.random_state)),
                'DecisionTree': OneVsRestClassifier(DecisionTreeClassifier(random_state=self.random_state)),
                'XGBoost': OneVsRestClassifier(XGBClassifier(random_state=self.random_state)),
                'GradientBoosting': OneVsRestClassifier(GradientBoostingClassifier(random_state=self.random_state)),
                'LogisticRegression': OneVsRestClassifier(LogisticRegression(max_iter=500, penalty=None, solver='saga',
                                                         random_state=self.random_state)),
                'LassoRegression': OneVsRestClassifier(LogisticRegression(max_iter=500, penalty='l1', solver='saga',
                                                      random_state=self.random_state)),
                'RidgeRegression': OneVsRestClassifier(LogisticRegression(max_iter=500, penalty='l2', solver='saga',
                                                      random_state=self.random_state)),
                'LDA': OneVsRestClassifier(LinearDiscriminantAnalysis(store_covariance = True)),
                'QDA': OneVsRestClassifier(QuadraticDiscriminantAnalysis(store_covariance = True))
            }

            grid_param_grids = {
                'RandomForest': {'estimator__n_estimators': [50, 100, 200],
                                 'estimator__max_depth': [None, 10, 20],
                                 'estimator__min_samples_split': [2, 5, 10]},
                'DecisionTree': {'estimator__max_depth': [None, 10, 20],
                                 'estimator__min_samples_split': [2, 5, 10]},
                'XGBoost': {'estimator__n_estimators': [50, 100, 200],
                            'estimator__max_depth': [3, 6, 10],
                            'estimator__learning_rate': [0.01, 0.1, 0.2]},
                'GradientBoosting': {'estimator__n_estimators': [50, 100, 200],
                                     'estimator__max_depth': [3, 6, 10],
                                     'estimator__learning_rate': [0.01, 0.1, 0.2]},
                'LogisticRegression': {},
                'LassoRegression': {'estimator__C': [0.1, 1, 10]},
                'RidgeRegression': {'estimator__C': [0.1, 1, 10]},
                'LDA': {'estimator__solver': ['svd', 'lsqr', 'eigen']},
                'QDA': {'estimator__reg_param': [0, 0.001, 0.01]}
            }

            random_param_grids = {
                'RandomForest': {'estimator__n_estimators': list(range(1, 501)),
                                 'estimator__max_depth': list(range(1, 11)),
                                 'estimator__min_samples_split': list(range(2, 11))},
                'DecisionTree': {'estimator__max_depth': list(range(1, 11)),
                                 'estimator__min_samples_split': list(range(2, 11))},
                'XGBoost': {'estimator__n_estimators': list(range(1, 501)),
                            'estimator__max_depth': list(range(1, 11)),
                            'estimator__learning_rate': loguniform(1e-10, 1)},
                'GradientBoosting': {'estimator__n_estimators': list(range(1, 501)),
                                     'estimator__max_depth': list(range(1, 11)),
                                     'estimator__learning_rate': loguniform(1e-10, 1)},
                'LogisticRegression': {},
                'LassoRegression': {'estimator__C': loguniform(1e-10, 1e+10)},
                'RidgeRegression': {'estimator__C': loguniform(1e-10, 1e+10)},
                'LDA': {'estimator__solver': ['svd', 'lsqr', 'eigen']},
                'QDA': {'estimator__reg_param': loguniform(1e-10, 1)}
            }

        elif self.problem_type == 'regression':
            models = {
                'RandomForest': RandomForestRegressor(random_state=self.random_state),
                'DecisionTree': DecisionTreeRegressor(random_state=self.random_state),
                'XGBoost': XGBRegressor(random_state=self.random_state),
                'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state),
                'LinearRegression': LinearRegression()
            }

            grid_param_grids = {
                'RandomForest': {'n_estimators': [50, 100, 200],
                                 'max_depth': [None, 10, 20],
                                 'min_samples_split': [2, 5, 10]},
                'DecisionTree': {'max_depth': [None, 10, 20],
                                 'min_samples_split': [2, 5, 10]},
                'XGBoost': {'n_estimators': [50, 100, 200],
                            'max_depth': [3, 6, 10],
                            'learning_rate': [0.01, 0.1, 0.2]},
                'GradientBoosting': {'n_estimators': [50, 100, 200],
                                      'max_depth': [3, 6, 10],
                                      'learning_rate': [0.01, 0.1, 0.2]},
                'LinearRegression': {}
            }

            random_param_grids = {
                'RandomForest': {'n_estimators': list(range(1, 501)),
                                  'max_depth': list(range(1, 11)),
                                  'min_samples_split': list(range(2, 11))},
                'DecisionTree': {'max_depth': list(range(1, 11)),
                                  'min_samples_split': list(range(2, 11))},
                'XGBoost': {'n_estimators': list(range(1, 501)),
                            'max_depth': list(range(1, 11)),
                            'learning_rate': loguniform(1e-10, 1)},
                'GradientBoosting': {'n_estimators': list(range(1, 501)),
                                      'max_depth': list(range(1, 11)),
                                      'learning_rate': loguniform(1e-10, 1)},
                'LinearRegression': {}
            }

        else:
            raise ValueError("Unknown problem type. Choose 'binary_classification', 'multiclass_classification' or 'regression'.")

        return models, grid_param_grids, random_param_grids

    def _define_scoring(self):
        """
        Defining the metric for model evaluation.

        Returns:
            available_metrics.get(self.metric) (str): Name of the metric.
        """
        if self.problem_type == 'binary_classification':
            available_metrics = {
                'roc_auc': 'roc_auc',
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'balanced_accuracy': 'balanced_accuracy'
            }
            if self.metric is None:
                self.metric = 'accuracy'

        elif self.problem_type == 'multiclass_classification':
            available_metrics = {
                'roc_auc': make_scorer(roc_auc_score, multi_class='ovr', response_method='predict_proba'),
                'accuracy': 'accuracy',
                'precision': make_scorer(precision_score, average='macro', zero_division=0),
                'recall': make_scorer(recall_score, average='macro'),
                'f1': make_scorer(f1_score, average='macro'),
                'balanced_accuracy': 'balanced_accuracy'
            }
            if self.metric is None:
                self.metric = 'accuracy'

        elif self.problem_type == 'regression':
            available_metrics = {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error'
            }
            if self.metric is None:
                self.metric = 'r2'

        else:
            raise ValueError("Unknown problem type. Choose 'binary_classification', 'multiclass_classification' or 'regression'.")

        return available_metrics.get(self.metric), available_metrics

    def _optimize(self, X, y, models, grid_param_grids, random_param_grids, eval_scoring, scoring):
        """
        Selecting, training and optimizing models.

        Params:
            X (DataFrame): Preprocessed input data;
            y (Series): Preprocessed target variable;
            models (dict): Models to be trained;
            grid_param_grids (dict): Grid of parameters for grid search optimization;
            random_param_grids (dict): Grid of parameters for random search optimization;
            eval_scoring (str): Metric used for model evaluation;
            scoring (dict): Additional metrics used in training process.
        """
        self.results = []
        self.scores = {}
        self.explainers = []
        self.optimizers = {}

        self.best_model, self.best_score = None, -float('inf')

        for model_name, model in models.items():
            print(f"Training model: {model_name}")
            if self.method == 'grid_search':
                optimizer = GridSearchCV(estimator=model, param_grid=grid_param_grids[model_name], cv=self.cv_folds,
                                         scoring=scoring, refit=eval_scoring)
            elif self.method == 'random_search':
                optimizer = RandomizedSearchCV(estimator=model, param_distributions=random_param_grids[model_name],
                                               n_iter=self.n_iter,
                                               cv=self.cv_folds, random_state=self.random_state, scoring=scoring,
                                               refit=eval_scoring)
            else:
                raise ValueError("Unknown optimization method. Choose 'grid_search' or 'random_search'.")

            start_time = time.time()
            optimizer.fit(X, y)
            end_time = time.time()
            training_time = end_time - start_time

            self.results.append(
                (model_name, optimizer.best_params_, optimizer.best_score_, eval_scoring, training_time))
            self.optimizers[model_name] = optimizer.best_estimator_
            self.explainers.append(dx.Explainer(optimizer, X.astype(float), y, label=model_name))

            if self.problem_type in ['binary_classification', 'multiclass_classification']:
                Metric = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']
                Result = [optimizer.cv_results_['mean_test_roc_auc'][optimizer.best_index_],
                          optimizer.cv_results_['mean_test_accuracy'][optimizer.best_index_],
                          optimizer.cv_results_['mean_test_precision'][optimizer.best_index_],
                          optimizer.cv_results_['mean_test_recall'][optimizer.best_index_],
                          optimizer.cv_results_['mean_test_f1'][optimizer.best_index_],
                          optimizer.cv_results_['mean_test_balanced_accuracy'][optimizer.best_index_]]
                self.scores[model_name] = pd.DataFrame({'Metric': Metric, 'Result': Result})

            elif self.problem_type == 'regression':
                Metric = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
                Result = [optimizer.cv_results_['mean_test_r2'][optimizer.best_index_],
                          optimizer.cv_results_['mean_test_neg_mean_squared_error'][optimizer.best_index_],
                          optimizer.cv_results_['mean_test_neg_mean_absolute_error'][optimizer.best_index_]]
                self.scores[model_name] = pd.DataFrame({'Metric': Metric, 'Result': Result})

            score = optimizer.best_score_
            if score > self.best_score:
                self.best_model = model_name
                self.best_estimator = optimizer.best_estimator_
                self.best_params = optimizer.best_params_
                self.best_score = optimizer.best_score_
                self.eval_scoring = eval_scoring
                self.training_time = training_time

    def _generate_report(self, X, y):
        """
        Generating report with model results including:
            - Best model results,
            - All results,
            - Descriptive statistics for numerical variables,
            - Histogram for target variable,
            - Classes breakdown for target variable,
            - Permutation feature importance,
            - Partial dependence profiles (PDP),
            - Accumulated local dependence profiles (ALE),
            - Break Down plots (for regression),
            - Ceteris Paribus plots (for regression),
            - Reports based on specifics of each model.

        Params:
            X (DataFrame): Preprocessed input data;
            y (Series): Preprocessed target variable.
        """
        print("\nReport with model results:")
        print("----------------------------------")

        # Best model results
        self._best_model_results()

        # All results
        self._all_results()

        # Descriptive statistics for numerical variables
        self._descriptive_statistics()

        # Histogram for target variable
        self._histogram(y)

        # Classes breakdown for target variable
        self._classes_breakdown()

        # Permutation feature importance
        self._permutation_feature_importance()

        # Partial dependence profiles (PDP)
        self._pdp()

        # Accumulated local effects (ALE)
        self._ale()

        # Finding observations with the largest and smallest prediction difference from actual values for the best model for regression
        if self.problem_type == 'regression':
            diff = self.best_estimator.predict(X) - y
            max_diff = np.where(np.max(np.abs(diff)) == np.abs(diff))[0].tolist()
            min_diff = np.where(np.min(np.abs(diff)) == np.abs(diff))[0].tolist()

            # Break Down plots
            self._break_down(X, max_diff, min_diff)

            # Ceteris Paribus plots
            self._ceteris_paribus(X, max_diff, min_diff)

        # Reports based on specifics of each model
        for name in self.all_results['Model_Name']:
            self._generate_individual_report(name)

    def _best_model_results(self):
        """
        Printing basic results for the best model.
        """
        print(f"Best model: {self.best_model}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Metric: {self.eval_scoring}")
        print(f"Training time: {self.training_time:.2f} seconds")
        print("----------------------------------")

    def _all_results(self):
        """
        Printing results for models chosen during parameter optimization and training process.
        """
        print("Results:")
        columns = ["Model_Name", "Best_Params", "Best_Score", "Eval_Scoring", "Training_Time"]
        self.all_results = pd.DataFrame(self.results, columns=columns)
        self.all_results['Best_Score'] = self.all_results['Best_Score'].map(lambda x: f"{x:.4f}")
        self.all_results['Training_Time'] = self.all_results['Training_Time'].map(lambda x: f"{x:.2f} seconds")
        print(self.all_results)
        print("----------------------------------")

    def _descriptive_statistics(self):
        """
        Printing basic descriptive statistics for numerical variables.
        """
        print("Basic descriptive statistics for numerical variables:")
        print(self.summary)
        print("----------------------------------")

    def _histogram(self, y):
        """
        Plotting histogram for target variable for regression.

        Params:
            y (Series): Target variable.
        """
        if self.problem_type == 'regression':
            print("Histogram for target variable:")
            y.hist()
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title('Histogram of y')
            plt.show()
            print("----------------------------------")

    def _classes_breakdown(self):
        """
        Printing breakdown of the counts for each class for target variable for classification.
        """
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            print("Breakdown of the counts for each class:")
            print(self.class_counts)
            print("----------------------------------")

    def _permutation_feature_importance(self):
        """
        Permutation feature importance for all models.
        """
        print("Permutation feature importance:")
        model_parts = []
        for explainer in self.explainers:
            model_parts.append(explainer.model_parts())
        print(model_parts[0].plot(model_parts[1:]))
        print("----------------------------------")

    def _pdp(self):
        """
        Partial dependence profiles (PDP) for all models.
        """
        print("Partial dependence profiles (PDP):")
        model_partials = []
        for explainer in self.explainers:
            model_partials.append(explainer.model_profile(type='partial', label=explainer.label))
        print(model_partials[0].plot(model_partials[1:]))
        print("----------------------------------")

    def _ale(self):
        """
        Accumulated local dependence profiles (ALE) for all models.
        """
        print("Accumulated local dependence profiles (ALE):")
        model_accumulated = []
        for explainer in self.explainers:
            model_accumulated.append(explainer.model_profile(type='accumulated', label=explainer.label))
        print(model_accumulated[0].plot(model_accumulated[1:]))
        print("----------------------------------")

    def _break_down(self, X, max_diff, min_diff):
        """
        Plotting Break Down plots for observations with the largest and smallest prediction difference from actual values for the best model for regression task.
        Params:
            X (DataFrame): Preprocessed input data;
            max_diff (list): List of indices of observations with the largest prediction difference from actual values for the best model
            min_diff (list): List of indices of observations with the smallest prediction difference from actual values for the best model
        """
        if self.problem_type == 'regression':
            print("Break Down plots for observations with the largest prediction difference from actual values for the best model:")
            print("----------------------------------")

            for i in max_diff:
                print(f"Break Down plot for observation no. {i}:")
                break_down = []
                for explainer in self.explainers:
                    break_down.append(explainer.predict_parts(X.iloc[i, :], type="break_down"))
                print(break_down[0].plot(break_down[1:]))
                print("----------------------------------")

            print("Break Down plots for observations with the smallest prediction difference from actual values for the best model:")
            print("----------------------------------")

            for i in min_diff:
                print(f"Break Down plot for observation no. {i}:")
                break_down = []
                for explainer in self.explainers:
                    break_down.append(explainer.predict_parts(X.iloc[i, :], type="break_down"))
                print(break_down[0].plot(break_down[1:]))
                print("----------------------------------")

    def _ceteris_paribus(self, X, max_diff, min_diff):
        """
        Plotting Ceteris Paribus plots for observations with the largest and smallest prediction difference from actual values for the best model for regression task.
        Params:
            X (DataFrame): Preprocessed input data;
            max_diff (list): List of indices of observations with the largest prediction difference from actual values for the best model
            min_diff (list): List of indices of observations with the smallest prediction difference from actual values for the best model
        """
        if self.problem_type == 'regression':
            print("Ceteris Paribus plots for observations with the largest prediction difference from actual values for the best model:")
            print("----------------------------------")

            for i in max_diff:
                print(f"Ceteris Paribus plot for observation no. {i}:")
                ceteris_paribus = []
                for explainer in self.explainers:
                    ceteris_paribus.append(explainer.predict_profile(X.iloc[i, :]))
                print(ceteris_paribus[0].plot(ceteris_paribus[1:]))
                print("----------------------------------")

            print("Ceteris Paribus plots for observations with the smallest prediction difference from actual values for the best model:")
            print("----------------------------------")

            for i in min_diff:
                print(f"Ceteris Paribus plot for observation no. {i}:")
                ceteris_paribus = []
                for explainer in self.explainers:
                    ceteris_paribus.append(explainer.predict_profile(X.iloc[i, :]))
                print(ceteris_paribus[0].plot(ceteris_paribus[1:]))
                print("----------------------------------")

    def _generate_individual_report(self, model_name = None):
        """
        Generate additional report based on specifics of a chosen model.

        Parameters:
            model_name (str): Name of model, if None, best model on training data is used;
        """
        if model_name is None:
            model_name = self.best_model

        # Generating report for decision tree model
        self._generate_DecisionTree_report(model_name)

        # Generating report for random forest model
        self._generate_RandomForest_report(model_name)

        # Generating report for gradient boosting model
        self._generate_GradientBoosting_report(model_name)

        # Generating report for XGBoost model
        self._generate_XGBoost_report(model_name)

        # Generating report for logistic/lasso/ridge regression model
        self._generate_LogisticLassoRidgeRegression_report(model_name)

        # Generating report for linear discriminant analysis model
        self._generate_LDA_report(model_name)

        # Generating report for quadratic discriminant analysis model
        self._generate_QDA_report(model_name)

        # Generating report for linear regression model
        self._generate_LinearRegression_report(model_name)

    def _generate_DecisionTree_report(self, model_name):
        """
        Generating report for decision tree model including:
            - Scores for decision tree,
            - Textual representation of the tree,
            - Decision tree visualization,
            - Feature importance plots.

        Params:
            model_name (str): Name of model.
        """
        if model_name == 'DecisionTree':
            print(f"\nRaport with scores for model {model_name}:")
            print("----------------------------------")

            # Basic scores for decision tree
            print(f"Model: {model_name}")
            print(f"Parameters: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Params'].iloc[0]}")
            print(f"Score: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Score'].iloc[0]}")
            print(f"Evaluation Metric: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Eval_Scoring'].iloc[0]}")
            print(f"Training Time: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Training_Time'].iloc[0]}")
            print("----------------------------------")

            # Scores for decision tree
            print(self.scores[model_name])
            print("----------------------------------")

            # Textual representation of the tree
            print("Textual representation of the tree:")
            if self.problem_type == 'multiclass_classification':
                for i, estimator in enumerate(self.optimizers[model_name].estimators_):
                    print(f"Rules for class {self.unique_values[i]}:")
                    tree_rules = export_text(estimator, feature_names=self.final_features)
                    print(tree_rules)
                    print("----------------------------------")

            else:
                tree_rules = export_text(self.optimizers[model_name], feature_names=self.final_features)
                print(tree_rules)
                print("----------------------------------")

            # Decision tree visualization
            print("Decision tree visualization:")
            if self.problem_type == 'multiclass_classification':
                for i, estimator in enumerate(self.optimizers[model_name].estimators_):
                    plt.figure(figsize=(12, 8))
                    print(f"Plotting tree for class {self.unique_values[i]}:")
                    plot_tree(estimator, feature_names=self.final_features, class_names=self.unique_values,
                              filled=True)
                    plt.title(f"Decision Tree for class {self.unique_values[i]}")
                    plt.show()

            elif self.problem_type == 'binary_classification':
                plt.figure(figsize=(12, 8))
                plot_tree(self.optimizers[model_name], feature_names=self.final_features, class_names=self.unique_values, filled=True)
                plt.title("Decision Tree")
                plt.show()

            elif self.problem_type == 'regression':
                plt.figure(figsize=(12, 8))
                plot_tree(self.optimizers[model_name], feature_names = self.final_features,
                          filled=True, rounded=True)
                plt.show()
            print("----------------------------------")

            # Feature importance
            print("Feature Importance:")
            if self.problem_type == 'multiclass_classification':
                all_importances = []
                for i, estimator in enumerate(self.optimizers[model_name].estimators_):
                    all_importances.append(estimator.feature_importances_)
                feature_importances = np.mean(all_importances, axis=0)

            else:
                feature_importances = self.optimizers[model_name].feature_importances_

            plt.bar(self.final_features, feature_importances)
            plt.xticks(rotation=90)
            plt.title(f"Feature Importances for {model_name}")
            plt.show()
            print("----------------------------------")

    def _generate_RandomForest_report(self, model_name):
        """
        Generating report for random forest model including:
            - Scores for random forest,
            - Visualization of the first decision tree,
            - Feature importance plots.

        Params:
            model_name (str): Name of model.
        """
        if model_name == 'RandomForest':
            print(f"\nRaport with scores for model {model_name}:")
            print("----------------------------------")

            # Basic scores for random forest
            print(f"Model: {model_name}")
            print(f"Parameters: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Params'].iloc[0]}")
            print(f"Score: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Score'].iloc[0]}")
            print(f"Evaluation Metric: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Eval_Scoring'].iloc[0]}")
            print(f"Training Time: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Training_Time'].iloc[0]}")
            print("----------------------------------")

            # Scores for random forest
            print(self.scores[model_name])
            print("----------------------------------")

            # Visualization of the first decision tree
            print("Visualization of the first decision tree:")
            if self.problem_type == 'multiclass_classification':
                for i, estimator in enumerate(self.optimizers[model_name].estimators_):
                    first_tree = estimator.estimators_[0]
                    plt.figure(figsize=(12, 8))
                    plot_tree(first_tree, feature_names=self.final_features, class_names=self.unique_values,
                        filled=True, rounded=True)
                    plt.title(f"Tree visualization for class: {self.unique_values[i]}")
                    plt.show()

            elif self.problem_type == 'binary_classification':
                plt.figure(figsize=(12, 8))
                plot_tree(self.optimizers[model_name].estimators_[0], feature_names=self.final_features,
                          class_names=self.unique_values, filled=True)
                plt.show()

            else:
                plt.figure(figsize=(12, 8))
                plot_tree(self.optimizers[model_name].estimators_[0], feature_names=self.final_features,
                          filled=True, rounded=True)
                plt.show()
            print("----------------------------------")

            # Feature importance plots
            print("Feature Importance:")
            if self.problem_type == 'multiclass_classification':
                all_importances = []
                for i, estimator in enumerate(self.optimizers[model_name].estimators_):
                    all_importances.append(estimator.feature_importances_)
                feature_importances = np.mean(all_importances, axis=0)

            else:
                feature_importances = self.optimizers[model_name].feature_importances_

            plt.bar(self.final_features, feature_importances)
            plt.xticks(rotation=90)
            plt.title(f"Feature Importances for {model_name}")
            plt.show()
            print("----------------------------------")

    def _generate_GradientBoosting_report(self, model_name):
        """
        Generating report for gradient boosting model including:
            - Scores for gradient boosting,
            - Visualization of the first decision tree,
            - Feature importance plots.

        Params:
            model_name (str): Name of model.
        """
        if model_name == 'GradientBoosting':
            print(f"\nRaport with scores for model {model_name}:")
            print("----------------------------------")

            # Basic scores for gradient boosting
            print(f"Model: {model_name}")
            print(f"Parameters: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Params'].iloc[0]}")
            print(f"Score: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Score'].iloc[0]}")
            print(f"Evaluation Metric: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Eval_Scoring'].iloc[0]}")
            print(f"Training Time: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Training_Time'].iloc[0]}")
            print("----------------------------------")

            # Scores for gradient boosting
            print(self.scores[model_name])

            # Visualization of the first decision tree
            print("----------------------------------")
            print("Visualization of the first decision tree:")
            if self.problem_type == 'multiclass_classification':
                for class_idx, estimator in enumerate(self.optimizers[model_name].estimators_):
                    tree = estimator.estimators_[0, 0]
                    plt.figure(figsize=(12, 8))
                    plot_tree(tree, feature_names=self.final_features, class_names=self.unique_values, filled=True, rounded=True)
                    plt.title(f"First Tree for Class {self.unique_values[class_idx]}")
                    plt.show()

            elif self.problem_type == 'binary_classification':
                plt.figure(figsize=(12, 8))
                plot_tree(self.optimizers[model_name].estimators_[0][0], feature_names=self.final_features,
                          class_names=self.unique_values, filled=True, rounded=True)
                plt.show()

            else:
                plt.figure(figsize=(12, 8))
                plot_tree(self.optimizers[model_name].estimators_[0][0], feature_names=self.final_features,
                          filled=True, rounded=True)
                plt.show()
            print("----------------------------------")

            # Feature importance plots
            print("Feature Importance:")
            if self.problem_type == 'multiclass_classification':
                all_importances = []
                for i, estimator in enumerate(self.optimizers[model_name].estimators_):
                    all_importances.append(estimator.feature_importances_)
                feature_importances = np.mean(all_importances, axis=0)

            else:
                feature_importances = self.optimizers[model_name].feature_importances_

            plt.bar(self.final_features, feature_importances)
            plt.xticks(rotation=90)
            plt.title(f"Feature Importances for {model_name}")
            plt.show()
            print("----------------------------------")

    def _generate_XGBoost_report(self, model_name):
        """
        Generating report for XGBoost model including:
            - Scores for XGBoost,
            - Feature importance plots.

        Params:
            model_name (str): Name of model.
        """
        if model_name == 'XGBoost':
            print(f"\nRaport with scores for model {model_name}:")
            print("----------------------------------")

            # Basic scores for XGBoost
            print(f"Model: {model_name}")
            print(f"Parameters: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Params'].iloc[0]}")
            print(f"Score: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Score'].iloc[0]}")
            print(f"Evaluation Metric: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Eval_Scoring'].iloc[0]}")
            print(f"Training Time: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Training_Time'].iloc[0]}")
            print("----------------------------------")

            # Scores for XGBoost
            print(self.scores[model_name])
            print("----------------------------------")

            # Feature importance plots
            print("Feature Importance:")
            if self.problem_type == 'multiclass_classification':
                all_importances = []
                for i, estimator in enumerate(self.optimizers[model_name].estimators_):
                    all_importances.append(estimator.feature_importances_)
                feature_importances = np.mean(all_importances, axis=0)

            else:
                feature_importances = self.optimizers[model_name].feature_importances_

            plt.bar(self.final_features, feature_importances)
            plt.xticks(rotation=90)
            plt.title(f"Feature Importances for {model_name}")
            plt.show()
            print("----------------------------------")

    def _generate_LogisticLassoRidgeRegression_report(self, model_name):
        """
        Generating report for logistic/lasso/ridge regression model including:
            - Scores for logistic/lasso/ridge regression,
            - Model coefficients,
            - Visualization of model coefficients.

        Params:
            model_name (str): Name of model.
        """
        if model_name in ['LogisticRegression', 'LassoRegression', 'RidgeRegression']:
            print(f"\nRaport with scores for model {model_name}:")
            print("----------------------------------")

            # Basic scores for logistic/lasso/ridge regression
            print(f"Model: {model_name}")
            print(f"Parameters: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Params'].iloc[0]}")
            print(f"Score: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Score'].iloc[0]}")
            print(f"Evaluation Metric:: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Eval_Scoring'].iloc[0]}")
            print(f"Training Time: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Training_Time'].iloc[0]}")
            print("----------------------------------")

            # Scores for logistic/lasso/ridge regression
            print(self.scores[model_name])
            print("----------------------------------")

            # Model coefficients with visualization
            print("Model coefficients:")
            if self.problem_type == 'multiclass_classification':
                for class_idx, estimator in enumerate(self.optimizers[model_name].estimators_):
                    coefficients = estimator.coef_.flatten()
                    intercept = estimator.intercept_.flatten()
                    print(f"Coefficients for class {self.unique_values[class_idx]}:")
                    for feature, coef in zip(self.final_features, coefficients):
                        print(f"  Feature: {feature},  Coefficient: {coef}")
                    print(f"Intercept for class {self.unique_values[class_idx]}: {intercept[0]}")
                    print("----------------------------------")

                    print("Visualization of coefficients:")
                    plt.figure(figsize=(10, 6))
                    plt.barh(self.final_features, coefficients, color='skyblue')
                    plt.xlabel('Coefficient Value')
                    plt.title(f'Visualization of coefficients for class {self.unique_values[class_idx]}')
                    plt.show()

            else:
                coefficients = self.optimizers[model_name].coef_
                intercept = self.optimizers[model_name].intercept_
                for class_idx, class_coef in enumerate(coefficients):
                    print(f"Coefficients for class {self.unique_values[class_idx]}:")
                    for feature, coef in zip(self.final_features, class_coef):
                        print(f"  Feature: {feature},  Coefficient: {coef}")
                    print(f"Intercept for class {self.unique_values[class_idx]}: {intercept[class_idx]}")
                print("----------------------------------")

                print("Visualization of coefficients:")
                for class_idx, class_coef in enumerate(coefficients):
                    plt.figure(figsize=(10, 6))
                    plt.barh(self.final_features, class_coef, color='skyblue')
                    plt.xlabel('Coefficient Value')
                    plt.title(f'Visualization of coefficients for class {self.unique_values[class_idx]}')
                    plt.show()
            print("----------------------------------")

    def _generate_LDA_report(self, model_name):
        """
        Generating report for linear discriminant analysis model including:
            - Scores for linear discriminant analysis,
            - Model coefficients,
            - Visualization of model coefficients,
            - Means and covariance matrix.

        Params:
            model_name (str): Name of model.
        """
        if model_name == 'LDA':
            print(f"\nRaport with scores for model {model_name}:")
            print("----------------------------------")

            # Basic scores for LDA
            print(f"Model: {model_name}")
            print(f"Parameters: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Params'].iloc[0]}")
            print(f"Score: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Score'].iloc[0]}")
            print(f"Evaluation Metric:: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Eval_Scoring'].iloc[0]}")
            print(f"Training Time: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Training_Time'].iloc[0]}")
            print("----------------------------------")

            # Scores for LDA
            print(self.scores[model_name])
            print("----------------------------------")

            # Model coefficients with visualization
            print("Model coefficients:")
            if self.problem_type == 'multiclass_classification':
                for class_idx, estimator in enumerate(self.optimizers[model_name].estimators_):
                    coefficients = estimator.coef_.flatten()
                    intercept = estimator.intercept_.flatten()
                    print(f"Coefficients for class {self.unique_values[class_idx]}:")
                    for feature, coef in zip(self.final_features, coefficients):
                        print(f"  Feature: {feature},  Coefficient: {coef}")
                    print(f"Intercept for class {self.unique_values[class_idx]}: {intercept[0]}")
                    print("----------------------------------")

                    print("Visualization of coefficients:")
                    plt.figure(figsize=(10, 6))
                    plt.barh(self.final_features, coefficients, color='skyblue')
                    plt.xlabel('Coefficient Value')
                    plt.title(f'Visualization of coefficients for class {self.unique_values[class_idx]}')
                    plt.show()

            else:
                coefficients = self.optimizers[model_name].coef_
                intercept = self.optimizers[model_name].intercept_
                for class_idx, class_coef in enumerate(coefficients):
                    print(f"Coefficients for class {self.unique_values[class_idx]}:")
                    for feature, coef in zip(self.final_features, class_coef):
                        print(f"  Feature: {feature},  Coefficient: {coef}")
                    print(f"Intercept for class {self.unique_values[class_idx]}: {intercept[class_idx]}")
                print("----------------------------------")

                print("Visualization of coefficients:")
                for class_idx, class_coef in enumerate(coefficients):
                    plt.figure(figsize=(10, 6))
                    plt.barh(self.final_features, class_coef, color='skyblue')
                    plt.xlabel('Coefficient Value')
                    plt.title(f'Visualization of coefficients for class {self.unique_values[class_idx]}')
                    plt.show()
            print("----------------------------------")

            # Means and covariance matrix
            print("Means and covariance matrix:")
            if self.problem_type == 'multiclass_classification':
                for class_idx, estimator in enumerate(self.optimizers[model_name].estimators_):
                    means = estimator.means_
                    covariance = estimator.covariance_
                    print(f"Class {self.unique_values[class_idx]}:")
                    print(f"Means: {means}")
                    print(f"Covariance matrix: {covariance}")
                    print("----------------------------------")

            else:
                means = self.optimizers[model_name].means_
                covariances = self.optimizers[model_name].covariance_
                print("Means:", means)
                print("Covariance matrix:", covariances)
                print("----------------------------------")

    def _generate_QDA_report(self, model_name):
        """
        Generating report for quadratic discriminant analysis model including:
            - Scores for quadratic discriminant analysis,
            - Means and covariance matrix.

        Params:
            model_name (str): Name of model.
        """
        if model_name == "QDA":
            print(f"\nRaport with scores for model {model_name}:")
            print("----------------------------------")

            # Basic scores for QDA
            print(f"Model: {model_name}")
            print(f"Parameters: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Params'].iloc[0]}")
            print(f"Score: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Score'].iloc[0]}")
            print(f"Evaluation Metric: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Eval_Scoring'].iloc[0]}")
            print(f"Training Time: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Training_Time'].iloc[0]}")
            print("----------------------------------")

            # Scores for QDA
            print(self.scores[model_name])
            print("----------------------------------")

            # Means and covariance matrix
            print("Means and covariance matrix:")
            if self.problem_type == 'multiclass_classification':
                for class_idx, estimator in enumerate(self.optimizers[model_name].estimators_):
                    means = estimator.means_
                    covariance = estimator.covariance_
                    print(f"Class {self.unique_values[class_idx]}:")
                    print(f"Means: {means}")
                    print(f"Covariance matrix: {covariance}")
                    print("----------------------------------")

            else:
                means = self.optimizers[model_name].means_
                covariances = self.optimizers[model_name].covariance_
                print("Means:", means)
                print("Covariance matrix:", covariances)
                print("----------------------------------")

    def _generate_LinearRegression_report(self, model_name):
        """
        Generating report for linear regression model including:
            - Scores for linear regression,
            - Model coefficients,
            - Visualization of model coefficients.

        Params:
            model_name (str): Name of model.
        """
        if model_name == 'LinearRegression':
            print(f"\nRaport with scores for model {model_name}:")
            print("----------------------------------")

            # Basic scores for linear regression
            print(f"Model: {model_name}")
            print(f"Parameters: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Params'].iloc[0]}")
            print(f"Score: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Best_Score'].iloc[0]}")
            print(f"Evaluation Metric: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Eval_Scoring'].iloc[0]}")
            print(f"Training Time: {self.all_results.loc[self.all_results['Model_Name'] == model_name, 'Training_Time'].iloc[0]}")
            print("----------------------------------")

            # Scores for linear regression
            print(self.scores[model_name])
            print("----------------------------------")

            # Model coefficients
            print("Model coefficients:")
            coefficients = self.optimizers[model_name].coef_
            intercept = self.optimizers[model_name].intercept_
            coef_df = pd.DataFrame(coefficients, columns=['Coefficient'], index=self.final_features)
            print(coef_df)
            print(f"Intercept: {intercept}")
            print("----------------------------------")

            # Visualization of coefficients
            print("Visualization of coefficients:")
            plt.barh(self.final_features, coefficients)
            plt.xlabel("Coefficients")
            plt.title("Linear Regression Coefficients")
            plt.show()
            print("----------------------------------")

    def _preprocess_test_data(self, X):
        """
        Preprocessing of test dataset including:
            - Converting input data to DataFrame,
            - Splitting columns to different types,
            - Imputation of missing values,
            - Standardizing numerical data,
            - Encoding categorical data,
            - Selecting features chosen during training process.

        Parameters:
            X (array/DataFrame): Input data.

        Return:
             X_copy (DataFrame): Preprocessed input data.
        """

        # Converting input data
        X_copy = self._to_dataframe(X)

        # Splitting columns to different types
        X_copy, binary_cols, numerical_cols, categorical_cols, datetime_cols = self._categorize_columns(X_copy)

        # Imputation of missing values
        X_copy = self._impute_missing_values(X_copy, binary_cols, numerical_cols, categorical_cols)

        # Standardizing numerical data
        X_copy = self._standardize(X_copy, numerical_cols)

        # Encoding categorical data
        X_copy = self._encode(X_copy, categorical_cols)

        # Selecting features chosen during training process
        X_copy = X_copy[self.final_features]

        return X_copy

    def _preprocess_test_and_target_data(self, X, y):
        """
        Preprocessing of test dataset including:
            - Converting target variable to Series,
            - Converting input data to DataFrame,
            - Handling missing values in y,
            - Processing target variable,
            - Splitting columns to different types,
            - Imputation of missing values,
            - Standardizing numerical data,
            - Encoding categorical data,
            - Selecting features chosen during training process.

        Parameters:
            X (array/DataFrame): Input data;
            y (array/Series): Target variable.

        Return:
             X_copy (DataFrame): Preprocessed input data;
             y_copy (Series): Preprocessed target variable.
        """

        # Converting target variable
        y_copy = self._to_series(y)

        # Converting input data
        X_copy = self._to_dataframe(X)

        # Handling missing values in y
        X_copy, y_copy = self._remove_missing_y(X_copy, y_copy)

        # Processing target variable
        y_copy = self._process_y(y_copy)

        # Splitting columns to different types
        X_copy, binary_cols, numerical_cols, categorical_cols, datetime_cols = self._categorize_columns(X_copy)

        # Imputation of missing values
        X_copy = self._impute_missing_values(X_copy, binary_cols, numerical_cols, categorical_cols)

        # Standardizing numerical data
        X_copy = self._standardize(X_copy, numerical_cols)

        # Encoding categorical data
        X_copy = self._encode(X_copy, categorical_cols)

        # Selecting features chosen during training process
        X_copy = X_copy[self.final_features]

        return X_copy, y_copy

    def _generate_binary_classification_report(self, model_name, y_true, y_pred, y_pred_proba, prediction_threshold):
        """
        Generating report on the test data for binary classification including:
            - Metric values on the test set,
            - ROC Curve,
            - Confusion Matrix for chosen threshold.

        Parameters:
            model_name (str): Name of model;
            y_true (Series): Preprocessed target variable;
            y_pred (Series): Predictions;
            y_pred_proba (Series): Predictions of probabilities;
            prediction_threshold (float): Threshold for confusion matrix.
        """
        print(f"\nReport with scores for model {model_name} on the test set:")
        print("----------------------------------")

        # Metric values on the test set
        print("Metric values on the test set:")
        accuracy = np.round(accuracy_score(y_true, y_pred), 4)
        precision = np.round(precision_score(y_true, y_pred), 4)
        recall = np.round(recall_score(y_true, y_pred), 4)
        f1 = np.round(f1_score(y_true, y_pred), 4)
        balanced_accuracy = np.round(balanced_accuracy_score(y_true, y_pred), 4)
        auc_score = np.round(roc_auc_score(y_true, y_pred_proba[:, 1]), 4)
        test_results = {
            'Metric': ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'AUC'],
            'Score': [accuracy, precision, recall, f1, balanced_accuracy, auc_score]
        }
        test_results = pd.DataFrame(test_results)
        print(test_results)
        print("----------------------------------")

        # ROC Curve
        print("ROC Curve:")
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()
        print("----------------------------------")

        # Confusion matrix
        print(f"Confusion Matrix for threshold {prediction_threshold}:")
        if prediction_threshold == 0.5:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.unique_values,
                        yticklabels=self.unique_values)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()

        else:
            y_pred_to_proba = (y_pred_proba >= prediction_threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_to_proba[:, 1])
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.unique_values,
                        yticklabels=self.unique_values)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()
        print("----------------------------------")

        # Mean Gini and Mean Entropy
        print("Other metrics:")
        print(f"Mean Gini: {np.mean(1 - np.sum(y_pred_proba ** 2, axis=1)):.4f}")
        print(f"Mean Entropy: {np.mean(-np.sum(y_pred_proba * np.log2(y_pred_proba + 1e-9), axis=1)):.4f}")
        print("----------------------------------")

    def _generate_multiclass_classification_report(self, model_name, y_true, y_pred, y_pred_proba):
        """
        Generating report on the test data for multiclass classification including:
            - Metric values on the test set,
            - ROC Curve,
            - Confusion Matrix.

        Parameters:
            model_name (str): Name of model;
            y_true (Series): Preprocessed target variable;
            y_pred (Series): Predictions;
            y_pred_proba (Series): Predictions of probabilities.
        """
        print(f"\nReport with scores for model {model_name} on the test set:")
        print("----------------------------------")

        # Metric values on the test set
        print("Metric values on the test set:")
        accuracy = np.round(accuracy_score(y_true, y_pred), 4)
        precision = np.round(precision_score(y_true, y_pred, average='macro'), 4)
        recall = np.round(recall_score(y_true, y_pred, average='macro'), 4)
        f1 = np.round(f1_score(y_true, y_pred, average='macro'), 4)
        balanced_accuracy = np.round(balanced_accuracy_score(y_true, y_pred), 4)
        auc_score = np.round(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'), 4)
        test_results = {
            'Metric': ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'AUC'],
            'Score': [accuracy, precision, recall, f1, balanced_accuracy, auc_score]
        }
        test_results = pd.DataFrame(test_results)
        print(test_results)

        # ROC Curve
        print("----------------------------------")
        print("ROC Curve:")
        plt.figure(figsize=(10, 8))
        lw = 2
        colormap = get_cmap("tab10")
        for i in range(len(self.unique_values)):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            color = colormap(i / len(self.unique_values))
            plt.plot(fpr, tpr, color=color, lw=lw, label=f'ROC curve (class {i}) (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multiclass Classification (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.show()
        print("----------------------------------")

        # Confusion matrix
        print(f"Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.unique_values,
                    yticklabels=self.unique_values)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
        print("----------------------------------")

        # Mean Gini and Mean Entropy
        print("Other metrics:")
        print(f"Mean Gini: {np.mean(1 - np.sum(y_pred_proba ** 2, axis=1)):.4f}")
        print(f"Mean Entropy: {np.mean(-np.sum(y_pred_proba * np.log2(y_pred_proba + 1e-9), axis=1)):.4f}")
        print("----------------------------------")

    def _generate_regression_report(self, model_name, y_true, y_pred):
        """
        Generating report on the test data for binary classification including metric values on the test set.

        Parameters:
            model_name (str): Name of model;
            y_true (Series): Preprocessed target variable;
            y_pred (Series): Predictions.
        """
        print(f"\nReport with scores for model {model_name} on the test set:")
        print("----------------------------------")

        # Metric values on the test set
        print("Metric values on the test set:")
        r2 = np.round(r2_score(y_true, y_pred), 4)
        mse = np.round(mean_squared_error(y_true, y_pred), 4)
        mae = np.round(mean_absolute_error(y_true, y_pred), 4)
        test_results = {
            'Metric': ['r2', 'mse', 'mae'],
            'Score': [r2, mse, mae]
        }
        test_results = pd.DataFrame(test_results)
        print(test_results)
        print("----------------------------------")



