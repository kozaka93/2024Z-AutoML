import pandas as pd
from numerai_automl.data_managers.data_loader import DataLoader
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.model_managers.base_model_manager import BaseModelManager
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager
from numerai_automl.model_managers.meta_model_manager import MetaModelManager
from numerai_automl.raport_manager.raport_manager import RaportManager
from numerai_automl.scorer.scorer import Scorer
from numerai_automl.config.config import TARGET_CANDIDATES
from numerai_automl.visual.cumsum_cor_plot import CumSumCorPlot
from numerai_automl.visual.radar_plot import RadarPlot


def main():
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
    return_data["predictions_model_meta_weighted"] = predictor_weighted(X)
    return_data["predictions_model_meta_lgbm"] = predictor_lgbm(X)
    return_data["predictions_model_omega"] = (return_data[["predictions_model_meta_weighted", "predictions_model_meta_lgbm"]].sum(axis=1)) / 2

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

    print(f"FINISHED SCORE COMPUTING")

    df = return_data[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", "predictions_model_omega", 'era',  "target"]]
    csp = CumSumCorPlot(df)
    cumsum_cor_plot = csp.get_plot()
    df = pd.read_csv("return_data_for_scoring.csv", index_col=0)
    df = df.loc[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", "predictions_model_omega"]]
    rp = RadarPlot(df)
    radar_plot = rp.get_plot()
    rm = RaportManager([cumsum_cor_plot, radar_plot])
    rm.generate_html("raport.html")

    print("FINISHED GENERATING RAPORT")