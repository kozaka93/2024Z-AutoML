from matplotlib import pyplot as plt
import numpy as np
from numerai_automl.visual.abstract_plot import AbstractPlot


class FeatureExposuresPlot(AbstractPlot):
    """
    This class creates bar plot for each feature's values - lots of them.
    data_for_visualization: The data to be visualized. This should be a pandas DataFrame in the following format:
    - Index: The eras
    - Columns: The features
    - Values: The correlation of the features with the target - they will be shown as bars.
    """

    def get_plot(self) -> plt.Figure:
        feature_exposures = self.data_for_visualization
        axes = feature_exposures.plot.bar(
            title="Feature Exposures",
            figsize=(16, 10),
            layout=(7, 5),
            xticks=[],
            subplots=True,
            sharex=False,
            legend=False,
            snap=False
        )
        fig = axes[0, 0].figure if isinstance(axes, np.ndarray) else axes.figure
        for ax in fig.axes:
            ax.set_xlabel("")
            ax.title.set_fontsize(10)
        plt.tight_layout(pad=1.5)
        fig.suptitle("Feature Exposures", fontsize=15)
        return fig