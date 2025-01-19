
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.adapt import MLkNN
from python_files.Preprocess import PaddingEstimator, add_pad, extract_features_with_window, process_labels_with_window, WindowFeatureExtractor, WindowLabelProcessor, process_labels_with_window_2d, PCADimensionReducer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

feature_pipeline = Pipeline([
                ('pca', PCADimensionReducer()),
                ('imputer', SimpleImputer(strategy='mean')),  
                ('scaler', StandardScaler()), 
            ])

pipelines = {
                'OneVsRest_LogisticRegression': Pipeline([
                ('preprocessing', feature_pipeline),  
                ('model', OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=100)))  
                ]),
                'SVC': Pipeline([
                    ('preprocessing', feature_pipeline),
                    ('model', OneVsRestClassifier(SVC(probability=True)))
                ]),

                'XGBoost': Pipeline([
                    ('preprocessing', feature_pipeline),
                    ('model', OneVsRestClassifier(XGBClassifier(n_jobs=4, eval_metric='auc',
                                                                objective='binary:hinge', tree_method='hist')))
                ]),
                'XGBoost_MultiOutput': Pipeline([
                    ('preprocessing', feature_pipeline),
                    ('model',XGBClassifier(
                            n_jobs=4,
                            objective='binary:logistic',
                            tree_method='hist',
                            multi_strategy='multi_output_tree',
                            random_state=42
                    ))
                ])
            }
            




param_distributions = {
                'SVC': {
                'model__estimator__C': [0.1, 1, 10],     
                'model__estimator__kernel': ['linear', 'rbf'], 
                'model__estimator__gamma': ['scale', 'auto'],  
                        },

                'OneVsRest_LogisticRegression': {
                'model__estimator__C': [0.1, 1, 10, 100],  
                'model__estimator__penalty': ['l2'],      
                'model__estimator__solver': ['liblinear', 'saga'],  
                'model__estimator__max_iter': [10000],    
                        },
                'XGBoost': {
                    'model__estimator__n_estimators': [500, 1000, 1500],
                    # 'model__estimator__max_depth': [1, 5, 10, 15],
                    # 'model__estimator__learning_rate': [0.01, 0.1, 0.3],
                    # 'model__estimator__subsample': [0.8, 1.0],
                        },
                'XGBoost_MultiOutput':{
                    'model__n_estimators': [500, 1000, 1500],
                        },

                
                
            }


def display_model_info(model):
    """
    Wyświetla nazwę modelu i jego parametry w czytelnej formie.

    Args:
        model: Pipeline lub model, który ma być opisany.
    """
    try:
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            actual_model = model.named_steps['model']
        else:
            actual_model = model

        model_name = type(actual_model).__name__
        model_params = actual_model.get_params()

        print("---------------------------------------")
        print(f"Best Model: {model_name}")
        print("Model Parameters:")

        for param, value in model_params.items():
            print(f"  {param}: {value}")
        print("---------------------------------------")

    except Exception as e:
        print(f"Wystąpił błąd podczas wyświetlania informacji o modelu: {e}")
