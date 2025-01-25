from numerai_automl.feature_neutralizer.feature_neutralizer import FeatureNeutralizer
import pandas as pd
import numpy as np
from numerai_automl.data_managers.data_loader import DataLoader

def test_neutralizer():

    data_loader = DataLoader(data_version="v5.0", feature_set="small")

    data = data_loader.load_train_data()

    predictions = data["target"]

    # print(predictions)

    predictions = predictions + np.random.normal(0, 0.1, len(predictions))


    neutralizer = FeatureNeutralizer(all_features=[col for col in data.columns if "feature" in col], target_name="target")
    score = neutralizer.find_neutralization_features_and_proportions(data, predictions)

    print(score)

if __name__ == "__main__":
    test_neutralizer()
