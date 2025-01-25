from numerai_automl.visual.abstract_plot import AbstractPlot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LowHighStdPlot(AbstractPlot):
    """
    This class creates a line plot for the features with the lowest and highest standard deviation or sharpe ratio or max drawdown or mean of the correlation of the features with the target.
    data_for_visualization: The data to be visualized. This should be a pandas DataFrame in the following format:
    - Index: The eras
    - Columns: The features
    - Values: The correlation of the features with the target
    feature_metrics: The metrics of the features. This should be a pandas DataFrame in the following format:
    - Index: The features
    - mean column: The mean of the correlation of the features with the target
    - std column: The standard deviation of the correlation of the features with the target
    - sharpe column: The sharpe ratio of the correlation of the features with the target
    - max_drawdown column: The maximum drawdown of the correlation of the features with the target

    """

    def __init__(self, data_for_visualization: pd.DataFrame, feature_metrics: pd.DataFrame, which_highest_lowest='std'):
        super().__init__(data_for_visualization)
        self.feature_metrics = feature_metrics
        self.which_highest_lowest = which_highest_lowest
        self.possible_which_highest_lowest = ['std', 'sharpe', 'max_drawdown', 'mean']
        if self.which_highest_lowest not in self.possible_which_highest_lowest:
            raise ValueError(f"which_highest_lowest should be one of {self.possible_which_highest_lowest}")

    def get_plot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(15, 5))
        per_era_corr = self.data_for_visualization
        per_era_corr[[self.feature_metrics[self.which_highest_lowest].idxmin(),
                      self.feature_metrics[self.which_highest_lowest].idxmax()]].plot(
            ax=ax, title="Per-era Correlation of Features to the Target", xlabel="Era"
        )
        plt.legend(["lowest " + self.which_highest_lowest, "highest " + self.which_highest_lowest])
        return fig
