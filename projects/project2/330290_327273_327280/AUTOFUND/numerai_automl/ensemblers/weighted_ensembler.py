import json
from typing import Any, Dict, List, Union
import pandas as pd
import random
from numerai_automl.scorer.scorer import Scorer
import cloudpickle
from numerai_automl.utils.utils import get_project_root
class WeightedTargetEnsembler:
    def __init__(self, 
                 all_neutralized_prediction_features: List[str] = [],
                 target_name: str = 'target',
                 number_of_interations: int = 30,
                 max_number_of_prediction_features_for_ensemble: int = 5,
                 number_of_diffrent_weights_for_ensemble: int = 5
                 ):
        """
        Initialize the WeightedTargetEnsembler for combining multiple prediction models.
        
        Parameters:
            all_neutralized_prediction_features: List of feature names containing neutralized predictions
            target_name: Name of the target variable column
            number_of_interations: Number of ensemble combinations to try
            max_number_of_prediction_features_for_ensemble: Maximum number of features to include in ensemble
            number_of_diffrent_weights_for_ensemble: Number of different weight combinations to try per iteration
        """
        self.scorer = Scorer()
        self.all_neutralized_prediction_features = all_neutralized_prediction_features
        self.target_name = target_name
        self.neutralized_predictions_model_target = f"neutralized_predictions_model_{target_name}"
        self.number_of_interations = number_of_interations
        self.max_number_of_prediction_features_for_ensemble = max_number_of_prediction_features_for_ensemble
        self.number_of_diffrent_weights_for_ensemble = number_of_diffrent_weights_for_ensemble
        self.best_ensemble_features_and_weights = None
        self.project_root = get_project_root()
        self.best_ensemble_features_and_weights = {}

        if all_neutralized_prediction_features != []:
            if self.max_number_of_prediction_features_for_ensemble > len(self.all_neutralized_prediction_features):
                self.max_number_of_prediction_features_for_ensemble = len(self.all_neutralized_prediction_features)

    
    def find_ensemble_prediction_features_and_proportions(self, train_data: pd.DataFrame, metric: str="mean") -> Dict[str, Union[Dict, Dict]]:
        """
        Find optimal combination of prediction features and their weights for ensemble.
        
        Parameters:
            validation_data: DataFrame containing:
                - Neutralized prediction features
                - 'era' column for time-based validation
                - Target variable column
            metric: Optimization metric ('mean', 'std', 'sharpe', or 'max_drawdown')
        Returns:
            Dictionary containing:
                - neutralization_params: Best feature combination and weights
                - scores: Performance metrics for the best ensemble
                
                example:
                {
                    "neutralization_params": {
                        "neutralized_prediction_features": ["neutralized_predictions_model_target", "neutralized_predictions_model_target_victor_20"],
                        "weights": [0.6, 0.4]
                    },
                    "scores": {
                        "mean": 0.025,
                        "sharpe": 1.5, 
                        "std": 0.015,
                        "max_drawdown": 0.1
                    }
                }
        """

        # check if metric is valid  
        assert metric in ["mean", "std", "sharpe", "max_drawdown"], "The metric is not valid"

        # check if neutralized_predictions_model_target is in the train_data
        assert self.neutralized_predictions_model_target in train_data.columns, f"The feature {self.neutralized_predictions_model_target} is not in the validation data"

        # check if all_neutralized_prediction_features are in the train_data
        for feature in self.all_neutralized_prediction_features:
            assert feature in train_data.columns, f"The feature {feature} is not in the validation data"

        # check if era is in the train_data
        assert "era" in train_data.columns, "The validation data does not have an era column"

        # check if target_name is in the validation_data
        assert self.target_name in train_data.columns, f"The target {self.target_name} is not in the validation data"

        train_data = train_data[self.all_neutralized_prediction_features + ["era", self.target_name]]

        ensemble_features_and_weights = {}

        ensemble_features_and_weights[f"ensemble_predictions_{0}"] = {
            "neutralized_prediction_features": [self.neutralized_predictions_model_target],
            "weights": [1]
        }

        ensemble_predictions_df = train_data[["era", self.target_name]].copy()
        ensemble_predictions_df[f"ensemble_predictions_{0}"] = train_data[self.neutralized_predictions_model_target]

        prediction_features_without_main_prediction = self.all_neutralized_prediction_features.copy()
        prediction_features_without_main_prediction.remove(self.neutralized_predictions_model_target)


        for i in range(self.number_of_interations):
            number_of_prediction_features_for_ensemble = random.randint(1, self.max_number_of_prediction_features_for_ensemble - 1)
            prediction_features_for_ensemble = random.sample(prediction_features_without_main_prediction, number_of_prediction_features_for_ensemble)
            prediction_features_for_ensemble.append(self.neutralized_predictions_model_target)

            weights_for_ensemble = self._mean_weights(len(prediction_features_for_ensemble))
            weights_series = pd.Series(weights_for_ensemble, index=prediction_features_for_ensemble)
            ensemble_predictions_df[f"ensemble_predictions_{1 + i * self.number_of_diffrent_weights_for_ensemble}"] = (train_data[prediction_features_for_ensemble] * weights_series).sum(axis=1)
            
            ensemble_features_and_weights[f"ensemble_predictions_{1 + i * self.number_of_diffrent_weights_for_ensemble}"] = {
                "neutralized_prediction_features": prediction_features_for_ensemble,
                "weights": weights_for_ensemble
            }
            
            for j in range(1, self.number_of_diffrent_weights_for_ensemble):
                weights_for_ensemble = self._random_weights(len(prediction_features_for_ensemble))
                weights_series = pd.Series(weights_for_ensemble, index=prediction_features_for_ensemble)
                ensemble_predictions_df[f"ensemble_predictions_{1 + i * self.number_of_diffrent_weights_for_ensemble + j}"] = (train_data[prediction_features_for_ensemble] * weights_series).sum(axis=1)
                
                ensemble_features_and_weights[f"ensemble_predictions_{1 + i * self.number_of_diffrent_weights_for_ensemble + j}"] = {
                    "neutralized_prediction_features": prediction_features_for_ensemble,
                    "weights": weights_for_ensemble
                }

        scores = self.scorer.compute_scores(ensemble_predictions_df, self.target_name)

        if metric == "mean" or metric == "sharpe":
            scores = scores.sort_values(by=metric, ascending=False)
        elif metric == "std" or metric == "max_drawdown":
            scores = scores.sort_values(by=metric, ascending=True)

        best_prediction_column = scores.index[0]
        best_ensemble_features_and_weights = ensemble_features_and_weights[best_prediction_column]
        best_scores = scores.loc[best_prediction_column].to_dict()

        self.best_ensemble_features_and_weights = best_ensemble_features_and_weights
        
        self.save_ensemble_model()
        
        return {
            "ensemble_features_and_weights": best_ensemble_features_and_weights,
            "scores": best_scores
        }
    
    def load_ensemble_features_and_weights(self):
        with open(f"{self.project_root}/models/ensemble_models/weighted_ensembler/weighted_ensembler_params.json", "r") as f:
            weighted_ensembler_params = json.load(f)
        self.best_ensemble_features_and_weights = weighted_ensembler_params["ensemble_features_and_weights"]
        return self.best_ensemble_features_and_weights

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Get the ensemble prediction for the train data
        """


        assert self.best_ensemble_features_and_weights is not None, "The ensemble features and weights are not loaded"
        assert "neutralized_prediction_features" in self.best_ensemble_features_and_weights, "The ensemble features and weights do not contain neutralized_prediction_features"
        assert "weights" in self.best_ensemble_features_and_weights, "The ensemble features and weights do not contain weights"

        for feature in self.best_ensemble_features_and_weights["neutralized_prediction_features"]:
            assert feature in X.columns, f"The feature {feature} is not in the data"

        weights = pd.Series(
            self.best_ensemble_features_and_weights["weights"],
            index=self.best_ensemble_features_and_weights["neutralized_prediction_features"]
        )

        return (X[self.best_ensemble_features_and_weights["neutralized_prediction_features"]] * weights).sum(axis=1)

    def save_ensemble_model(self):
        assert self.best_ensemble_features_and_weights is not None, "The ensemble features and weights are not loaded"
        
        # save model with pickl
        with open(f"{self.project_root}/models/ensemble_models/weighted_ensembler/weighted_ensembler.pkl", "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load_ensemble_model(cls):
        project_root = get_project_root()
        with open(f"{project_root}/models/ensemble_models/weighted_ensembler/weighted_ensembler.pkl", "rb") as f:
            loaded_instance = cloudpickle.load(f)
            return loaded_instance

    def _mean_weights(self, number_of_weights: int) -> List[float]:
        """
        Generate equally distributed weights that sum to 1.
        
        Parameters:
            number_of_weights: Number of weights to generate
        
        Returns:
            List of equal weights (1/n for each weight)
        """
        return [1 / number_of_weights for _ in range(number_of_weights)]
    
    def _random_weights(self, number_of_weights: int) -> List[float]:
        """
        Generate random weights that sum to 1 for ensemble experimentation.
        
        Parameters:
            number_of_weights: Number of weights to generate
        
        Returns:
            List of normalized random weights that sum to 1
        """

        arr = [random.uniform(0, 1) for _ in range(number_of_weights)]
        return [ weight / sum(arr) for weight in arr]


    