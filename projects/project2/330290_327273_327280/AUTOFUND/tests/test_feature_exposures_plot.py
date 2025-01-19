import pandas as pd
import numpy as np
from numerai_automl.visual.feature_exposures_plot import FeatureExposuresPlot


def test_feature_exposures_plot():
    df = pd.DataFrame(
        {'era': [i for i in range(1, 31)], "feature1": np.random.normal(0, 1, 30),
            "feature2": np.random.normal(0, 1, 30), "feature3": np.random.normal(0, 1, 30),
            "feature4": np.random.normal(0, 1, 30), "feature5": np.random.normal(0, 1, 30),
            "feature6": np.random.normal(0, 1, 30), "feature7": np.random.normal(0, 1, 30),
            "feature8": np.random.normal(0, 1, 30), "feature9": np.random.normal(0, 1, 30),
            "feature10": np.random.normal(0, 1, 30), "feature11": np.random.normal(0, 1, 30),
            "feature12": np.random.normal(0, 1, 30), "feature13": np.random.normal(0, 1, 30),
            "feature14": np.random.normal(0, 1, 30), "feature15": np.random.normal(0, 1, 30),
            "feature16": np.random.normal(0, 1, 30), "feature17": np.random.normal(0, 1, 30),
            "feature18": np.random.normal(0, 1, 30), "feature19": np.random.normal(0, 1, 30),
         })
    df = df.set_index("era")
    df.index.name = "era"
    fep = FeatureExposuresPlot(df)
    fig = fep.get_plot()
    fig.show()


if __name__ == "__main__":
    test_feature_exposures_plot()
