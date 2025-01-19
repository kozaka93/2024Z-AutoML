from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_models(seed):
    return {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(random_state=seed),
        'Random Forest': RandomForestClassifier(random_state=seed),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed),
        'SVM': SVC(probability=True)
    }

def get_param_grids():
    return {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear', 'saga']
        },
        'Decision Tree': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': [None, 'sqrt', 'log2']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [5, 10, 20, 50, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3, 4]
        },
    }
