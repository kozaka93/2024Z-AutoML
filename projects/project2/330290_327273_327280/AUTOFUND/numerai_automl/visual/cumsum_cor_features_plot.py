from numerai_automl.visual.abstract_plot import AbstractPlot
import matplotlib.pyplot as plt
import numpy as np

class CumsumCorrFeaturesPlot(AbstractPlot):
    """
    This class creates a line plot based on the data given.
    Data should be in the form of a pandas DataFrame in the following format:
    - Index: The eras
    - Columns: The features
    - Values: The correlation of the features with the target
    This class will plot the cumulative sum of the absolute value of the correlation of the features with the target.
    """

    def get_plot(self) -> plt.Figure:
        per_era_corr = self.data_for_visualization
        per_era_corr *= np.sign(per_era_corr.mean())
        fig = per_era_corr.cumsum().plot(
            title="Cumulative Absolute Value CORR of Features and the Target",
            figsize=(15, 5),
            legend=False,
            xlabel="Era"
        ).get_figure()
        return fig