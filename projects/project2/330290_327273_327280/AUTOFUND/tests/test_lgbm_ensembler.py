from numerai_automl.data_managers.data_loader import DataLoader
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.model_managers.base_model_manager import BaseModelManager
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager
from numerai_automl.model_managers.meta_model_manager import MetaModelManager
from numerai_automl.scorer.scorer import Scorer



def test_lgbm_ensembler():
    data_loader = DataLoader(data_version="v5.0", feature_set="medium")
    data_manager = DataManager(data_version="v5.0", feature_set="medium")


    ensemble_manager = EnsembleModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    # ensemble_manager.find_weighted_ensemble()
    ensemble_manager.find_lgbm_ensemble(number_of_iterations=2, cv_folds=5)


    

    


if __name__ == "__main__":
    test_lgbm_ensembler()

