from numerai_automl.visual.abstract_plot import AbstractPlot
from numerai_automl.scorer.scorer import Scorer
import matplotlib.pyplot as plt
import pandas as pd


class CumSumCorPlot(AbstractPlot):
    """
    This class creates a plot of the cumulative sum of the correlation between predictions and target for each era and model.
    Data should be in the form of a pandas DataFrame in the following format:
    - era Column: The era of the data.
    - predictions Columns: The predicted values for each model - these HAVE TO be named 'prediction_model_name'.
    - target Column: The actual target values.
    The Data will changed by Scorer in this class and then used to create the plot.
    """

    def __init__(self, data_for_visualization: pd.DataFrame, target_name: str = 'target'):
        super().__init__(data_for_visualization)
        self.data_for_visualization = Scorer().compute_cumsum_correlation_per_era(self.data_for_visualization,
                                                                                  target_name)

    def get_plot(self) -> plt.Figure:
        df = self.data_for_visualization
        df.columns = [col.replace('prediction_', '') for col in df.columns]
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(ax=ax, linewidth=2)
        ax.set_title('Cumulative Correlation of Predictions and Target for each Era', fontsize=16)
        ax.set_xlabel('Era', fontsize=14)
        ax.set_ylabel('Cumulative Correlation', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Model', fontsize=12)
        return fig
