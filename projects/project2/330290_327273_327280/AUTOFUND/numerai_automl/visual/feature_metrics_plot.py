from matplotlib import pyplot as plt

from numerai_automl.visual.abstract_plot import AbstractPlot


class FeatureMetricsPlot(AbstractPlot):
    """
    This class creates a bar subplot based on the data given.
    Data should be in the form of a pandas DataFrame in the following format:
    - Index: The names of Features to be compared
    - mean Column: The mean of the metric for each feature
    - std Column: The standard deviation of the metric for each feature
    - sharpe Column: The sharpes ratio of the metric for each feature
    - max_drawdown Column: The maximum drawdown of the metric for each feature
    - delta Column: The delta of the metric for each feature
    """

    def get_plot(self) -> plt.Figure:
        feature_metrics = self.data_for_visualization
        fig = feature_metrics.sort_values("mean", ascending=False).plot.bar(
            title="Performance Metrics of Features Sorted by Mean",
            subplots=True,
            figsize=(15, 6),
            layout=(2, 3),
            sharex=False,
            xticks=[],
            snap=False
        ).flatten()[0].get_figure()
        return fig
