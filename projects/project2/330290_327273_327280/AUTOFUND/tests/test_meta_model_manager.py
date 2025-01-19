from matplotlib import pyplot as plt
from numerai_automl.data_managers import data_manager
from numerai_automl.ensemblers.weighted_ensembler import WeightedTargetEnsembler
from numerai_automl.model_managers.meta_model_manager import MetaModelManager
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.scorer.scorer import Scorer


def test_meta_model_manager():


    model_manager = MetaModelManager(
        targets_names_for_base_models=["target", "target_victor_20"],
        )

    data_manager = DataManager(data_version="v5.0", feature_set="small")    

    # X = data_manager.load_live_data()
    X = data_manager.load_validation_data_for_ensembler()


    # TODO: THIS FUCNTION SHOULD BNE TOTALY DIFFENT NOW IT RETURNS ONLY PREDICTIONS
    # pred = model_manager.create_weighted_meta_model_predictions(X)

    # model_manager.save_predictor("weighted")

    func = model_manager.load_predictor("weighted")

    preds = func(X)
    print(preds)

    return preds

def test_meta_model_manager2():


    model_manager = MetaModelManager(
        targets_names_for_base_models=["target", "target_victor_20"],
        )

    data_manager = DataManager(data_version="v5.0", feature_set="small")  

    # model_manager.find_lgbm_ensemble()

    model_manager.save_predictor("lgbm")

    # X = data_manager.load_live_data()
    X = data_manager.load_validation_data_for_ensembler()




    func = model_manager.load_predictor("lgbm")

    preds = func(X)
    print(preds)

    return preds


    
if __name__ == "__main__":
    preds1 = test_meta_model_manager()
    preds2 = test_meta_model_manager2()


    data_manager = DataManager(data_version="v5.0", feature_set="small")  
    X = data_manager.load_validation_data_for_ensembler()

    X["meta_weighted_predictions"] = preds1
    X["meta_lgbm_predictions"] = preds2

    scorer = Scorer()

    scores = scorer.compute_scores(X, "target")

    print(scores)

    print(X.head())



