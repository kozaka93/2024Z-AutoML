import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from optuna.samplers import RandomSampler, TPESampler
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'experiment_{current_time}.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

RANDOM_SEEDS = [42, 69, 420]
RANDOM_SEED = RANDOM_SEEDS[0]
DATASET_IDS = [31, 37, 45062, 43980]
SAMPLERS = [RandomSampler, TPESampler]

lr_params = {
    'C': (1e-4, 1e4, 'log'),
    'penalty': (['elasticnet'], 'categorical'),
    'l1_ratio': (1e-4, 1.0, 'log'),
    'class_weight': (['balanced'], 'categorical'),
    'max_iter': (1500, 1500, 'int'),
    'solver': (['saga'], 'categorical')
}

tree_params = {
    'n_estimators': (10, 1000, 'int'),
    'criterion': (['gini', 'entropy', 'log_loss'], 'categorical'),
    'bootstrap': ([True], 'categorical'),
    'max_samples': (0.5, 1, 'float'),
    'max_features': (0.1, 0.9, 'float'),
    'min_samples_leaf': (0.05, 0.25, 'float'),
}

xgb_params = {
    'n_estimators': (10, 2000, 'int'),
    'learning_rate': (1e-4, 0.4, 'log'),
    'subsample': (0.25, 1.0, 'float'),
    'booster': (['gbtree'], 'categorical'),
    'max_depth': (1, 15, 'int'),
    'min_child_weight': (1, 128, 'float'),
    'colsample_bytree': (0.2, 1.0, 'float'),
    'colsample_bylevel': (0.2, 1.0, 'float'),
    'reg_alpha': (1e-4, 512.0, 'log'),
    'reg_lambda': (1e-3, 1e3, 'log')
}

MODELS = [
    (LogisticRegression, lr_params),
    (ExtraTreesClassifier, tree_params),
    (XGBClassifier, xgb_params)
]
