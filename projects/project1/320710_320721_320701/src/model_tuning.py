from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import loguniform
from skopt.space import Real, Categorical

def create_rf_pipeline(preprocessor):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

def create_lr_pipeline(preprocessor):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=5000, random_state=42))
    ])

def create_xg_pipeline(preprocessor):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

def perform_grid_search(pipeline, param_grid, X_train, y_train):
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def perform_random_search(pipeline, param_distributions, X_train, y_train, n_iter=20):
    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=n_iter, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search

def perform_bayesian_search(pipeline, search_spaces, X_train, y_train, n_iter=20):
    bayesian_search = BayesSearchCV(pipeline, search_spaces, n_iter=n_iter, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    bayesian_search.fit(X_train, y_train)
    return bayesian_search

def get_rf_search_spaces():
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    search_spaces = {
        'classifier__n_estimators': (100, 300),
        'classifier__max_depth': (1, 15),
        'classifier__min_samples_split': (2, 10),
        'classifier__min_samples_leaf': (1, 5)
    }
    return param_grid, search_spaces

def get_lr_search_spaces():
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'saga']
    }
    search_spaces = {
        'classifier__C': Real(0.01, 10, prior='log-uniform'), 
        'classifier__penalty': Categorical(['l2']),   
        'classifier__solver': Categorical(['lbfgs', 'saga'])  
    }
    return param_grid, search_spaces

def get_xg_search_spaces():
    param_grid = {
        'classifier__n_estimators': [100, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.2],
        'classifier__subsample': [0.6, 1.0],
    }
    search_spaces = {
        'classifier__n_estimators': (100, 300),
        'classifier__max_depth': (3, 8),
        'classifier__learning_rate': (0.01, 0.3, 'log-uniform'),
        'classifier__subsample': (0.5, 1.0),
    }
    return param_grid, search_spaces
