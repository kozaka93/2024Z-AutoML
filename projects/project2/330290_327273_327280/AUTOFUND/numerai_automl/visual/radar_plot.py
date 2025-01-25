from numerai_automl.visual.abstract_plot import AbstractPlot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


class RadarPlot(AbstractPlot):
    """
    This class creates a radar plot based on the data given.
    Data should be in the form of a pandas DataFrame in the following format:
    - Index: The names of Models to be compared, this will be the colors on the radar plot
    - mean Column: The mean of the metric for each model
    - std Column: The standard deviation of the metric for each model
    - sharpe Column: The sharpes ratio of the metric for each model
    - max_drawdown Column: The maximum drawdown of the metric for each model
    """

    def get_plot(self) -> plt.Figure:
        def normalize(column):
            minumum = column.min()
            maximum = column.max()
            return (column - minumum) / (maximum - minumum)
        print(self.data_for_visualization)
        df = self.data_for_visualization.apply(normalize)
        print(df)
        names_of_columns = df.columns.tolist()
        num_vars = len(names_of_columns)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        #
        # for i, label in enumerate(names_of_columns):
        #     values = df[label]
        #     min_val, max_val = values.min(), values.max()
        #     ax.set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))

        for index, row in df.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.fill(angles, values, alpha=0.1)
            ax.plot(angles, values, label=index)

        ax.set_ylim(bottom=df.min().min() - 0.05, top=df.max().max() + 0.05)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names_of_columns)
        plt.title("Radar Plot", y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        return fig
