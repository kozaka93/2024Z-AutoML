from matplotlib import pyplot as plt
from numerai_automl.data_managers import data_manager
from numerai_automl.ensemblers.weighted_ensembler import WeightedTargetEnsembler
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager
from numerai_automl.model_managers.meta_model_manager import MetaModelManager
from numerai_automl.data_managers.data_manager import DataManager


def test_ensemble_model_manager():


    model_manager = EnsembleModelManager(
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    model_manager.find_weighted_ensemble(max_number_of_prediction_features_for_ensemble=2)

    model_manager.find_lgbm_ensemble(max_number_of_prediction_features_for_ensemble=2)

def test_ensemble_model_manager2():
    model_manager = EnsembleModelManager(
        targets_names_for_base_models=["target", "target_victor_20"],
        )

    model_manager.find_lgbm_ensemble()

    
if __name__ == "__main__":
    test_ensemble_model_manager2()


