import json
from typing import Dict, List
from numerai_automl.config.config import MAIN_TARGET, TARGET_CANDIDATES, LIGHTGBM_PARAMS_OPTION, FEATURE_NEUTRALIZATION_PROPORTIONS
import os
import pandas as pd
import cloudpickle
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.ensemblers.weighted_ensembler import WeightedTargetEnsembler
from numerai_automl.feature_neutralizer.feature_neutralizer import FeatureNeutralizer
from numerai_automl.model_managers.base_model_manager import BaseModelManager
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager
from numerai_automl.model_trainers.lgbm_model_trainer import LGBMModelTrainer
from numerai_automl.utils.utils import get_project_root


target_candidates = TARGET_CANDIDATES
lightgbm_params = LIGHTGBM_PARAMS_OPTION
main_target = MAIN_TARGET
feature_neutralization_proportions = FEATURE_NEUTRALIZATION_PROPORTIONS

class MetaModelManager:
    """Manages meta-model operations including prediction neutralization and ensemble weighting.
    
    Args:
        data_version (str): Version of the dataset to use (default: "v5.0")
        feature_set (str): Feature set size to use (default: "small")
        targets_names_for_base_models (List[str]): List of target names for base models
        main_target (str): Main target for predictions
    """
    def __init__(self, data_version: str = "v5.0", 
                 feature_set: str = "small", 
                 targets_names_for_base_models: List[str] = target_candidates,
                 main_target: str = main_target):
        self.data_version = data_version
        self.feature_set = feature_set
        self.targets_names_for_base_models = targets_names_for_base_models
        self.data_manager = DataManager(data_version=data_version, feature_set=feature_set)
        self.features = self.data_manager.get_features()
        self.base_model_manager = BaseModelManager(
            data_version=data_version,
            feature_set=feature_set,
            targets_names_for_base_models=targets_names_for_base_models
        )
        self.ensemble_model_manager = EnsembleModelManager(
            data_version=data_version,
            feature_set=feature_set,
            targets_names_for_base_models=targets_names_for_base_models
        )
        self.feature_neutralizer = FeatureNeutralizer(all_features=self.features, target_name=main_target)
        self.project_root = get_project_root()

    def create_neutralized_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Creates neutralized predictions using base models and feature neutralization.
        
        Args:
            X (pd.DataFrame): Input features dataframe
            
        Returns:
            pd.DataFrame: DataFrame containing original and neutralized predictions
        """

        if "era" in X.columns:
            neutralized_predictions = X[self.features + ["era"]].copy()
        else:
            neutralized_predictions = X[self.features].copy()

        X = X[self.features]

        base_models = self.base_model_manager.load_base_models()
        neutralization_params = self.base_model_manager.load_neutralization_params()

        

        neutralized_predictions_names = []

        for target_name in self.targets_names_for_base_models:
            base_model = base_models[f"model_{target_name}"]

            preditions_name = f"predictions_model_{target_name}"


            neutralized_predictions[preditions_name] = base_model.predict(X)

            
            neutralized_predictions_names.append(f"neutralized_predictions_model_{target_name}")
            neutralized_predictions[f"neutralized_predictions_model_{target_name}"] = self.feature_neutralizer.apply_neutralization(neutralized_predictions, preditions_name, neutralization_params[preditions_name]["neutralization_params"])[f"neutralized_predictions_model_{target_name}"]

        if "era" in neutralized_predictions.columns:
            neutralized_predictions[neutralized_predictions_names] = neutralized_predictions[neutralized_predictions_names + ["era"]].groupby("era", group_keys=True).rank(pct=True)
        else:
            neutralized_predictions[neutralized_predictions_names] = neutralized_predictions[neutralized_predictions_names].rank(pct=True)


        return neutralized_predictions



    def create_weighted_meta_model_predictions(self, X: pd.DataFrame) -> pd.Series:
        weighted_ensembler = self.ensemble_model_manager.load_ensemble_model("weighted")
        neutralized_predictions = self.create_neutralized_predictions(X)

        predictions = weighted_ensembler.predict(neutralized_predictions)

        predictions.name = "meta_weighted_predictions"

        return predictions
    
    def create_lgbm_meta_model_predictions(self, X: pd.DataFrame) -> pd.Series:
        lgbm_ensembler = self.ensemble_model_manager.load_ensemble_model("lgbm")
        neutralized_predictions = self.create_neutralized_predictions(X)

        names = [f"neutralized_predictions_model_{target_name}" for target_name in self.targets_names_for_base_models]

        meta_predictions = lgbm_ensembler.predict(neutralized_predictions[names])
        
        if "era" in X.columns:
            predictions = neutralized_predictions[["era"]].copy()
            predictions["meta_lgbm_predictions"] = meta_predictions
            predictions = predictions.groupby("era", group_keys=True).rank(pct=True)
        else:
            predictions = pd.Series(meta_predictions).rank(pct=True)
            predictions.name = "meta_lgbm_predictions"

        
        return predictions

        

    def create_and_save_predictor(self, type: str):
        """Saves the meta model to disk."""

        if type == "weighted":
            predictions = self.create_weighted_meta_model_predictions
            # save model with pickl
            with open(f"{self.project_root}/models/meta_models/weighted_meta_model/weighted_meta_model.pkl", "wb") as f:
                cloudpickle.dump(predictions, f)
        elif type == "lgbm":
            predictions = self.create_lgbm_meta_model_predictions
            with open(f"{self.project_root}/models/meta_models/lgbm_meta_model/lgbm_meta_model.pkl", "wb") as f:
                cloudpickle.dump(predictions, f)
        else:
            raise ValueError(f"Unknown meta model type: {type}")

        

    def load_predictor(self, type: str):
        """Loads the meta model from disk."""
        if type == "weighted":
            with open(f"{self.project_root}/models/meta_models/weighted_meta_model/weighted_meta_model.pkl", "rb") as f:
                return cloudpickle.load(f)
        elif type == "lgbm":
            with open(f"{self.project_root}/models/meta_models/lgbm_meta_model/lgbm_meta_model.pkl", "rb") as f:
                return cloudpickle.load(f)
        else:
            raise ValueError(f"Meta model type {type} not found")