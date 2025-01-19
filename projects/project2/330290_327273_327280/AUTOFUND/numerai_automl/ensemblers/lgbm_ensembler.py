from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import KFold, ParameterSampler
from numerai_automl.model_trainers.lgbm_model_trainer import LGBMModelTrainer
from numerai_automl.scorer.scorer import Scorer
from numerai_automl.config.config import LIGHTGBM_PARAM_GRID, LIGHTGBM_PARAMS_OPTION
import cloudpickle
from numerai_tools.scoring import numerai_corr

from numerai_automl.utils.utils import get_project_root
from sklearn.model_selection import GroupKFold




lightgbm_param_grid = LIGHTGBM_PARAM_GRID

class LGBMEnsembler:
    def __init__(self, 
                 all_neutralized_prediction_features: List[str],
                 target_name: str = 'target',
                 number_of_iterations: int = 30,
                 cv_folds: int = 5
                 ):
        self.scorer = Scorer()
        self.all_neutralized_prediction_features = all_neutralized_prediction_features
        self.target_name = target_name
        self.neutralized_predictions_model_target = f"neutralized_predictions_model_{target_name}"
        self.model_trainer = None # LGBMModelTrainer(lightgbm_param_grid)
        self.project_root = get_project_root()
        self.number_of_iterations = number_of_iterations
        self.cv_folds = cv_folds
    
    # TODO: add metric to the function
    def find_lgbm_ensemble(self, train_data: pd.DataFrame, metric: str = "mean"):

        X = train_data[self.all_neutralized_prediction_features]
        y = train_data[self.target_name]
        eras = train_data["era"]

        # Initialize random parameter sampler
        param_list = list(ParameterSampler(
            lightgbm_param_grid, 
            n_iter=self.number_of_iterations
            # random_state=self.random_state
        ))

        best_score = float('-inf')
        best_params = None

        print(f"Starting Random Search with {self.number_of_iterations} iterations and {self.cv_folds}-fold cross-validation.")

        # Add base parameters to each sampled parameter set
        base_params = {
            'objective': 'regression',
            'verbose': -1,
            'min_split_gain': 0,  # Allow splits with smaller gains
            'min_data_in_leaf': 2,  # Allow smaller leaf nodes
            'boost_from_average': True
        }
        
        for idx, params in enumerate(param_list):
            # Merge base parameters with sampled parameters
            current_params = {**base_params, **params}
            
            cv_scores = []
            group_kf = GroupKFold(n_splits=self.cv_folds)
            for fold, (train_idx, val_idx) in enumerate(group_kf.split(X, y, groups=eras)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                eras_val = eras.iloc[val_idx]

                # Initialize and train the model with current params
                model = LGBMModelTrainer(current_params)
                model.train(X_train, y_train)
                predictions = model.get_model().predict(X_val)

                # Create a DataFrame for correlation calculation
                cv_data = pd.DataFrame({
                    'prediction': predictions,
                    'target': y_val,
                    'era': eras_val
                })


                
                # Calculate numerai correlation
                correlations = numerai_corr(cv_data[['prediction']], cv_data['target'])
                score = correlations.mean()

                cv_scores.append(score)

            avg_score = np.mean(cv_scores)
            print(f"Iteration {idx+1}/{self.number_of_iterations}: Avg CV Numerai Corr = {avg_score:.6f}")

            # Update best params
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        self.best_params = best_params
        self.best_score = best_score
        print(f"Best CV Numerai Corr: {self.best_score:.6f} with parameters: {self.best_params}")

        # Train the final model with best parameters on the entire dataset
        self.model_trainer = LGBMModelTrainer(self.best_params)
        self.model_trainer.train(X, y)
        self.model_trainer.get_model()

        self.save_ensemble_model()

    def predict(self, X: pd.DataFrame):
        return self.model_trainer.get_model().predict(X)
    
    def save_ensemble_model(self):
        assert self.model_trainer is not None, "The ensemble features and weights are not loaded"

        assert self.model_trainer.is_trained, "The model is not trained"
        
        # save model with pickl
        with open(f"{self.project_root}/models/ensemble_models/lgbm_ensembler/lgbm_ensembler.pkl", "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load_ensemble_model(cls):
        project_root = get_project_root()
        with open(f"{project_root}/models/ensemble_models/lgbm_ensembler/lgbm_ensembler.pkl", "rb") as f:
            return cloudpickle.load(f)