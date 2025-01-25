from matplotlib import pyplot as plt
from numerai_automl.ensemblers.weighted_ensembler import WeightedTargetEnsembler
from numerai_automl.model_managers.base_model_manager import BaseModelManager


def test_model_manager():
    model_manager = BaseModelManager(
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    model_manager.train_base_models()

    data_for_creating_predictions = model_manager.create_predictions_for_base_models()
    neutralization_params = model_manager.find_neutralization_features_and_proportions_for_base_models()
    print(neutralization_params)
    neutralized_predictions = model_manager.create_neutralized_predictions_from_base_models_predictions()
    print(neutralized_predictions)

def test_model_manager2():
    model_manager = BaseModelManager(
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    model_manager.load_base_models()
    neutralization_params = model_manager.find_neutralization_features_and_proportions_for_base_models()
    # print(neutralization_params)
    neutralized_predictions = model_manager.create_neutralized_predictions_from_base_models_predictions()

    # plot those predictions on the same plot
    print(neutralized_predictions[['target', 'neutralized_predictions_model_target', 'neutralized_predictions_model_target_victor_20']])


def test_model_manager3():
    model_manager = BaseModelManager(
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    model_manager.load_base_models()

    model_manager.load_neutralization_params()
    neutralized_predictions = model_manager.create_neutralized_predictions_from_base_models_predictions()

    # plot those predictions on the same plot
    print(neutralized_predictions.columns)

    print(neutralized_predictions[['target', 'neutralized_predictions_model_target', 'neutralized_predictions_model_target_victor_20']])


    model_manager.find_weighted_ensemble(
        metric="mean",
        target_name="target",
        number_of_iterations=2,
        max_number_of_prediction_features0_for_ensemble=2,
        number_of_diffrent_weights_for_ensemble=5
    )

    
if __name__ == "__main__":
    test_model_manager3()
