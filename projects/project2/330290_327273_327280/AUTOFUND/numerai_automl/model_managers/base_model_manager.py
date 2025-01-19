import json
from typing import Callable, Dict, List
from numerai_automl.config.config import MAIN_TARGET, TARGET_CANDIDATES, LIGHTGBM_PARAMS_OPTION, FEATURE_NEUTRALIZATION_PROPORTIONS
import os
import pandas as pd
import cloudpickle
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.ensemblers.weighted_ensembler import WeightedTargetEnsembler
from numerai_automl.feature_neutralizer.feature_neutralizer import FeatureNeutralizer
from numerai_automl.model_trainers.lgbm_model_trainer import LGBMModelTrainer
from numerai_automl.utils.utils import get_project_root

# find folder models in my project

target_candidates = TARGET_CANDIDATES
lightgbm_params = LIGHTGBM_PARAMS_OPTION
main_target = MAIN_TARGET
feature_neutralization_proportions = FEATURE_NEUTRALIZATION_PROPORTIONS


class BaseModelManager:
    """
    Manages the lifecycle of machine learning models for Numerai, including training, 
    prediction creation, and feature neutralization.

    This class handles:
    - Training and managing base models for different targets
    - Creating and saving predictions
    - Finding optimal feature neutralization parameters
    - Applying neutralization to predictions

    Attributes:
        project_root (str): Root path of the project
        targets_names_for_base_models (List[str]): List of target names to train models for
        data_manager (DataManager): Handles all data loading and saving operations
        lightgbm_params (Dict): Parameters for LightGBM models
        base_models (Dict): Dictionary storing trained models
        neutralization_params (Dict): Parameters for feature neutralization
    """

    def __init__(self, data_version: str = "v5.0", 
                 feature_set: str = "small", 
                 targets_names_for_base_models: List[str] = target_candidates,
                 lightgbm_params: Dict = lightgbm_params):
        """
        Initialize the ModelManager.

        Args:
            data_version (str): Version of the dataset to use
            feature_set (str): Size of feature set ('small' or 'medium')
            lightgbm_params (Dict): Parameters for LightGBM models
            targets_names_for_base_models (List[str]): List of targets to train models for
        """
        self.project_root = get_project_root()
        self.targets_names_for_base_models = targets_names_for_base_models
        self.data_manager = DataManager(data_version, feature_set)
        self.features_names = self.data_manager.get_features()
        self.lightgbm_params = lightgbm_params
        self.base_models = {}
        self.neutralization_params = {}

    def _get_features_names(self, data: pd.DataFrame) -> List[str]:
        """Extract feature column names from DataFrame."""
        return [col for col in data.columns if 'feature' in col]
    
    def _get_targets_names(self, data: pd.DataFrame) -> List[str]:
        """Extract target column names from DataFrame."""
        return [col for col in data.columns if 'target' in col]
    
    def _get_predictions_names(self, data: pd.DataFrame) -> List[str]:
        """Extract prediction column names from DataFrame."""
        return [col for col in data.columns if 'predictions' in col]

    def train_base_models(self) -> None:
        """
        Train base models for each specified target using LightGBM.
        Models are stored in self.base_models dictionary.
        """
        train_data = self.data_manager.load_train_data_for_base_models()
        features_names = self._get_features_names(train_data)
        targets_names = self._get_targets_names(train_data)

        assert all(target in targets_names for target in self.targets_names_for_base_models), \
            "Target names are not in targets columns"

        self.base_models = {}
        for target in self.targets_names_for_base_models:
            modelTrainer = LGBMModelTrainer(self.lightgbm_params)
            modelTrainer.train(train_data[features_names], train_data[target])
            self.base_models[f"model_{target}"] = modelTrainer.get_model()

    def save_base_models(self) -> None:
        """
        Save trained base models to disk using cloudpickle.
        Models are saved in the project's models/base_models directory.

        Raises:
            Exception: If no models exist to save
        """
        if self.base_models is None:
            raise Exception("Models do not exist")
        
        for model_name, model in self.base_models.items():
            model_path = f"{self.project_root}/models/base_models/{model_name}.pkl"
            with open(model_path, "wb") as f:
                cloudpickle.dump(model, f)

    def load_base_models(self) -> None:
        """
        Load previously saved base models from disk.
        Models are loaded from the project's models/base_models directory.
        """
        self.base_models = {}
        for target in self.targets_names_for_base_models:
            model_path = f"{self.project_root}/models/base_models/model_{target}.pkl"
            with open(model_path, "rb") as f:
                self.base_models[f"model_{target}"] = cloudpickle.load(f)

        return self.base_models

    def get_base_models(self) -> Dict:
        """
        Retrieve the dictionary of trained base models.

        Returns:
            Dict: Dictionary of trained models with keys as model names
        """
        return self.base_models
    
    def load_base_model_predictors(self) -> None:
        """
        Creates prediction functions for each base model that handle data preprocessing and prediction generation.

        This method:
        1. Loads base models if not already loaded
        2. Creates a predictor function for each model that:
           - Takes a DataFrame as input
           - Generates raw predictions using the model
           - Converts predictions to percentile ranks (overall or per era)
           - Returns a named pd.Series with predictions

        Returns:
            Dict[str, Callable]: Dictionary mapping model names to their predictor functions
                                Each predictor takes a DataFrame and returns a pd.Series
        """
        if self.base_models == {}:
            self.load_base_models()

        self.base_model_predictors = {}
        for model_name, model in self.base_models.items():
            
            def make_predictor(model, model_name) -> Callable[[pd.DataFrame], pd.Series]:
                def predict(data: pd.DataFrame) -> pd.Series:
                    
                    raw_predictions = model.predict(data[self.features_names])
                    predictions = pd.Series(raw_predictions, index=data.index)

                    if "era" in data.columns:
                        predictions = pd.Series(raw_predictions, index=data.index)
                        predictions = predictions.groupby(data["era"], group_keys=True).rank(pct=True)
                    else:
                        predictions = pd.Series(raw_predictions, index=data.index).rank(pct=True)
                    
                    predictions.name = f"predictions_{model_name}"
                    return predictions
                
                return predict

            self.base_model_predictors[f"{model_name}"] = make_predictor(model, model_name)

        return self.base_model_predictors

        
    def load_neutralized_base_model_predictors(self) -> None:
        """
        Creates prediction functions for each base model that include feature neutralization.

        This method:
        1. Loads base models and neutralization parameters if not already loaded
        2. Creates a predictor function for each model that:
           - Takes a DataFrame as input
           - Generates raw predictions
           - Applies feature neutralization using stored parameters
           - Converts predictions to percentile ranks (overall or per era)
           - Returns a named pd.Series with neutralized predictions

        The key difference from load_base_model_predictors is that these predictors
        apply feature neutralization to reduce the influence of specific features
        on the final predictions.

        Returns:
            Dict[str, Callable]: Dictionary mapping model names to their neutralized predictor functions
                                Each predictor takes a DataFrame and returns a pd.Series
        """

        if self.base_models == {}:
            self.load_base_models()
            
        if self.neutralization_params == {}:
            self.load_neutralization_params()

        feature_neutralizer = FeatureNeutralizer(self.features_names)

        def make_predictor(model, model_name, feature_neutralizer) -> Callable[[pd.DataFrame], pd.Series]:
            def predict(data: pd.DataFrame) -> pd.Series:
                model_predictions = model.predict(data[self.features_names])

                prediction_name = f"predictions_{model_name}"
                data_with_predictions = data.copy()
                data_with_predictions[prediction_name] = model_predictions

                neutralized_predictions = feature_neutralizer.apply_neutralization(
                    data=data_with_predictions,
                    prediction_name=prediction_name,
                    neutralization_params=self.neutralization_params[prediction_name]["neutralization_params"]
                )

                if "era" in data.columns:
                    neutralized_predictions[["era"]] = data[["era"]]
                    neutralized_predictions = neutralized_predictions.groupby("era", group_keys=True).rank(pct=True)
                else:
                    neutralized_predictions = neutralized_predictions.rank(pct=True)

                return neutralized_predictions[f"neutralized_{prediction_name}"]
            return predict

        self.neutralized_base_model_predictors = {}
        for model_name, model in self.base_models.items():
            self.neutralized_base_model_predictors[f"neutralized_{model_name}"] = make_predictor(
                model, model_name, feature_neutralizer
            )

        return self.neutralized_base_model_predictors

    
    def create_predictions_by_base_models(self) -> pd.DataFrame:
        """
        Create predictions using all base models on the latest tournament data.
        
        Returns:
            pd.DataFrame: DataFrame containing the original data plus predictions
                         from each base model in columns named 'predictions_model_{target}'
        """
        data_for_creating_predictions = self.data_manager.load_data_for_creating_predictions_for_base_models()
        features_names = self._get_features_names(data_for_creating_predictions)

        for target in self.targets_names_for_base_models:
            predictions = self.base_models[f"model_{target}"].predict(data_for_creating_predictions[features_names])
            data_for_creating_predictions[f"predictions_model_{target}"] = predictions

        self.data_manager.save_vanila_predictions_by_base_models(data_for_creating_predictions)
        return data_for_creating_predictions
    

    def find_neutralization_features_and_proportions_for_base_models(
            self,
            metric: str = "mean",
            target_name: str = main_target, 
            number_of_iterations: int = 10, 
            max_number_of_features_to_neutralize: int = 5, 
            proportions: List[float] = feature_neutralization_proportions
            ) -> Dict:
        """
        Find optimal feature neutralization parameters for each base model.

        Args:
            metric (str): Metric to optimize for ('mean', 'std', 'sharpe', 'max_drawdown')
            target_name (str): Name of the target to use for evaluation
            iterations (int): Number of random trials for finding optimal parameters
            max_number_of_features_to_neutralize (int): Maximum features to neutralize
            proportions (List[float]): List of neutralization proportions to try

        Returns:
            Dict: Neutralization parameters for each model
        """
        if self.base_models is None:
            raise Exception("Base models do not exist")
        
        assert metric in ["mean", "std", "sharpe", "max_drawdown"], "The metric is not valid"
        
        validation_data = self.data_manager.load_vanila_predictions_data_by_base_models()
        features_names = self._get_features_names(validation_data)
        predictions_names = self._get_predictions_names(validation_data)

        neutralizer = FeatureNeutralizer(
            features_names, 
            target_name, 
            number_of_iterations, 
            max_number_of_features_to_neutralize, 
            proportions
        )

        neutralization_params = {}
        for predictions_name in predictions_names:
            neutralization_params[predictions_name] = neutralizer.find_neutralization_features_and_proportions(
                validation_data[["era", target_name] + features_names],
                validation_data[predictions_name],
                metric
            )

        self.neutralization_params = neutralization_params
        self.save_neutralization_params()
        return neutralization_params
    
    def get_neutralization_params(self) -> Dict:
        """
        Retrieve the current neutralization parameters.

        Returns:
            Dict: Dictionary containing neutralization parameters for each model
        """
        return self.neutralization_params
    
    def save_neutralization_params(self) -> None:
        """
        Save neutralization parameters to disk in JSON format.
        Parameters are saved in the project's models/neutralization_params directory.

        Raises:
            Exception: If no neutralization parameters exist to save
        """
        if self.neutralization_params is None:
            raise Exception("Neutralization params do not exist")

        with open(f"{self.project_root}/models/neutralization_params/neutralization_params.json", "w") as f:
            json.dump(self.neutralization_params, f, indent=4)
    
    def load_neutralization_params(self) -> None:
        """
        Load previously saved neutralization parameters from disk.
        Parameters are loaded from the project's models/neutralization_params directory.
        """
        with open(f"{self.project_root}/models/neutralization_params/neutralization_params.json", "r") as f:
            self.neutralization_params = json.load(f)

        return self.neutralization_params
    
    def create_neutralized_predictions_by_base_models_predictions(self) -> pd.DataFrame:
        """
        Apply feature neutralization to the predictions from base models using stored
        neutralization parameters.

        The method:
        1. Loads vanilla (non-neutralized) predictions
        2. Applies neutralization using stored parameters
        3. Saves and returns the neutralized predictions

        Returns:
            pd.DataFrame: DataFrame containing the neutralized predictions for each model
                         in columns named 'neutralized_predictions_model_{target}'

        Raises:
            Exception: If neutralization parameters haven't been set
        """
        if self.neutralization_params is None:
            raise Exception("Neutralization params do not exist")
        
        vanila_predictions_data = self.data_manager.load_vanila_predictions_data_by_base_models()
        features_names = self._get_features_names(vanila_predictions_data)
        predictions_names = self._get_predictions_names(vanila_predictions_data)

        neutralized_predictions = vanila_predictions_data.copy()
        neutralizer = FeatureNeutralizer(features_names)

        for target in self.targets_names_for_base_models:
            pred_name = f"predictions_model_{target}"
            neutralized_predictions[f"neutralized_{pred_name}"] = neutralizer.apply_neutralization(
                vanila_predictions_data, 
                pred_name, 
                self.neutralization_params[pred_name]["neutralization_params"]
            )[f"neutralized_{pred_name}"]

        # Remove original prediction columns
        neutralized_predictions.drop(columns=predictions_names, inplace=True)
        self.data_manager.save_neutralized_predictions_by_base_models(neutralized_predictions)

        return neutralized_predictions
    
   


