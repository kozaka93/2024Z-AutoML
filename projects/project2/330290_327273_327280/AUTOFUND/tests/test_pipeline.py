import pandas as pd
from numerai_automl.data_managers.data_loader import DataLoader
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.model_managers.base_model_manager import BaseModelManager
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager
from numerai_automl.model_managers.meta_model_manager import MetaModelManager
from numerai_automl.scorer.scorer import Scorer
from numerai_automl.config.config import TARGET_CANDIDATES


def main_run():
    data_manager = DataManager(data_version="v5.0", feature_set="medium")
    model_manager = BaseModelManager(
        feature_set="medium",
        targets_names_for_base_models=TARGET_CANDIDATES,
        )
    model_manager.train_base_models()
    print("FINISHED TRAINING BASE MODELS")
    model_manager.save_base_models()
    print("FINISHED SAVING BASE MODELS")
    model_manager.create_predictions_by_base_models()
    print("FINISHED CREATING PREDICTIONS BY BASE MODELS")
    model_manager.find_neutralization_features_and_proportions_for_base_models(metric="sharpe", number_of_iterations=50, max_number_of_features_to_neutralize=120)
    print("FINISHED FINDING NEUTRALIZATION FEATURES AND PROPORTIONS FOR BASE MODELS")
    model_manager.create_neutralized_predictions_by_base_models_predictions()
    print("FINISHED CREATING NEUTRALIZED PREDICTIONS BY BASE MODELS PREDICTIONS")
    ensemble_manager = EnsembleModelManager(
        feature_set="medium",
        targets_names_for_base_models=TARGET_CANDIDATES,
        )
    ensemble_manager.find_weighted_ensemble(metric="sharpe", number_of_iterations=20, max_number_of_prediction_features_for_ensemble=12, number_of_diffrent_weights_for_ensemble=12)
    print("FINISHED FINDING WEIGHTED ENSEMBLE")
    ensemble_manager.find_lgbm_ensemble(number_of_iterations=20, cv_folds=5)
    print("FINISHED FINDING LGBM ENSEMBLE")
    meta_manager = MetaModelManager(
        feature_set="medium",
        targets_names_for_base_models=TARGET_CANDIDATES,
        )
    meta_manager.create_and_save_predictor("weighted")
    print("FINISHED CREATING AND SAVING WEIGHTED PREDICTOR")
    meta_manager.create_and_save_predictor("lgbm")
    print("FINISHED CREATING AND SAVING LGBM PREDICTOR")


    # now loading data for meta model
    X = data_manager.load_validation_data_for_meta_model()

    return_data = X.copy()

    print("FINISHED LOADING DATA FOR META MODEL")
    # now predicting
    predictor_weighted = meta_manager.load_predictor("weighted")
    predictor_lgbm = meta_manager.load_predictor("lgbm")
    return_data["predictions_meta_weighted"] = predictor_weighted(X)
    return_data["predictions_meta_lgbm"] = predictor_lgbm(X)

    base_models_predictors = model_manager.load_base_model_predictors()
    neutralized_base_models_predictors = model_manager.load_neutralized_base_model_predictors()

    for target_name in TARGET_CANDIDATES:
        return_data[f"predictions_model_{target_name}"] = base_models_predictors[f"model_{target_name}"](X)
        return_data[f"neutralized_predictions_model_{target_name}"] = neutralized_base_models_predictors[f"neutralized_model_{target_name}"](X)



    print("FINISHED PREDICTING")

    return_data.to_csv("return_data.csv")

    scorer = Scorer()
    scores = scorer.compute_scores(return_data, "target")

    scores.to_csv("return_data_for_scoring.csv")


def test_pipeline():
    data_manager = DataManager(data_version="v5.0", feature_set="small")
    model_manager = BaseModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    model_manager.train_base_models()

    print("FINISHED TRAINING BASE MODELS")

    model_manager.save_base_models()

    print("FINISHED SAVING BASE MODELS")

    model_manager.create_predictions_by_base_models()

    print("FINISHED CREATING PREDICTIONS BY BASE MODELS")

    model_manager.find_neutralization_features_and_proportions_for_base_models()

    model_manager.create_neutralized_predictions_by_base_models_predictions()

    ensemble_manager = EnsembleModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    ensemble_manager.find_weighted_ensemble()
    ensemble_manager.find_lgbm_ensemble()


    meta_manager = MetaModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    predictor_weighted = meta_manager.save_predictor("weighted")
    predictor_lgbm = meta_manager.save_predictor("lgbm")

    X = data_manager.load_validation_data_for_ensembler()

    preds_weighted = predictor_weighted(X)
    preds_lgbm = predictor_lgbm(X)

    X["meta_weighted_predictions"] = preds_weighted
    X["meta_lgbm_predictions"] = preds_lgbm

    scorer = Scorer()
    scores = scorer.compute_scores(X, "target")

    print(scores)

    print(X.head())

    print(preds_weighted)
    print(preds_lgbm)



def test_pipeline2():
    data_manager = DataManager(data_version="v5.0", feature_set="medium")
    model_manager = BaseModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    model_manager.load_base_models()

    model_manager.create_predictions_by_base_models()

    model_manager.find_neutralization_features_and_proportions_for_base_models()

    model_manager.create_neutralized_predictions_by_base_models_predictions()

    ensemble_manager = EnsembleModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    ensemble_manager.find_weighted_ensemble()
    ensemble_manager.find_lgbm_ensemble()


    meta_manager = MetaModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    meta_manager.save_predictor("weighted")
    meta_manager.save_predictor("lgbm")

    predictor_weighted = meta_manager.load_predictor("weighted")
    predictor_lgbm = meta_manager.load_predictor("lgbm")

    # this will be end validation data that we will do scoring plots etc staff like that.
    X = data_manager.load_validation_data_for_ensembler()



    preds_weighted = predictor_weighted(X)
    preds_lgbm = predictor_lgbm(X)

    X["meta_weighted_predictions"] = preds_weighted
    X["meta_lgbm_predictions"] = preds_lgbm

    scorer = Scorer()
    scores = scorer.compute_scores(X, "target")

    print(scores)

    print(X.head())

    print(preds_weighted)
    print(preds_lgbm)


def test_pipeline3():
    data_loader = DataLoader(data_version="v5.0", feature_set="medium")
    data_manager = DataManager(data_version="v5.0", feature_set="medium")
    meta_manager = MetaModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    model_manager = BaseModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    predictors = model_manager.load_base_model_predictors()
    target_predictor = predictors["model_target"]
    target_victor_20_predictor = predictors["model_target_victor_20"]

    neutralized_predictors = model_manager.load_neutralized_base_model_predictors()
    neutralized_target_predictor = neutralized_predictors["neutralized_model_target"]
    neutralized_target_victor_20_predictor = neutralized_predictors["neutralized_model_target_victor_20"]

    

    predictor_weighted = meta_manager.load_predictor("weighted")
    predictor_lgbm = meta_manager.load_predictor("lgbm")

    # this will be end validation data that we will do scoring plots etc staff like that.
    # X = data_loader.load_validation_data() # this i only checked to see comparison with notebooks
    X = data_manager.load_validation_data_for_meta_model()

    # take top 2000 rows
    # X = X.head(2000)

    features = data_manager.get_features()
    preds_target = target_predictor(X)
    preds_target_neutralized = neutralized_target_predictor(X)
    preds_target_victor_20 = target_victor_20_predictor(X[features])
    preds_target_victor_20_neutralized = neutralized_target_victor_20_predictor(X[features])
    preds_weighted = predictor_weighted(X)
    preds_lgbm = predictor_lgbm(X)


    X["predictions_model_target"] = preds_target
    X["neutralized_predictions_model_target"] = preds_target_neutralized
    X["predictions_model_target_victor_20"] = preds_target_victor_20
    X["neutralized_predictions_model_target_victor_20"] = preds_target_victor_20_neutralized
    print(preds_weighted)
    print(X)
    X["meta_weighted_predictions"] = preds_weighted
    X["meta_lgbm_predictions"] = preds_lgbm

    scorer = Scorer()
    scores = scorer.compute_scores(X, "target")

    print(scores)

    print(X.head())

    # print(preds_weighted)
    # print(preds_lgbm)


def test_pipeline4():
    X = pd.read_csv("return_data.csv")
    scorer = Scorer()
    scores = scorer.compute_scores(X, "target")

    # take rows: predictions_model_target, predictions_model_target_victor_20, meta_weighted_predictions, meta_lgbm_predictions
    # rows not columns
    # X = X[["predictions_model_target", "predictions_model_target_victor_20", "meta_weighted_predictions", "meta_lgbm_predictions"]]
    scores.to_csv("return_data_for_scoring.csv")
    print(scores)

if __name__ == "__main__":
    main_run()

