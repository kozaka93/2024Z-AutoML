
import pandas as pd
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.scorer.scorer import Scorer


def test_data_manager():
    data_manager = DataManager()

    data = data_manager.load_validation_data_for_neutralization_of_base_models()

    print(data.columns)
    print(data)

def test_data_manager_2():
    data_manager = DataManager()
    scorer = Scorer()

    data = data_manager.load_ranked_neutralized_predictions_for_base_models()
    print(scorer.compute_scores(data, "target"))
    print(data.columns)

    data2 = data_manager.load_neutralized_predictions_for_base_models()
    print(data2.columns)
    print(scorer.compute_scores(data2, "target"))



if __name__ == "__main__":
    test_data_manager_2()