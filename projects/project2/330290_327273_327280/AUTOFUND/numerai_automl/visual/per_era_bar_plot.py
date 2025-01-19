from numerai_automl.visual.abstract_plot import AbstractPlot
import matplotlib.pyplot as plt


class PerEraBarPlot(AbstractPlot):
    """
    This class get values for each era only and then it creates a bar plot for each era.
    Data should be in the form of a pandas DataFrame in the following format:
    - era Index: The era of the data.
    - values ONE Column: The values for each era to be plotted. - It will detect the column automatically.
    """

    def get_plot(self, values_column: str = None) -> plt.Figure:
        df = self.data_for_visualization
        if values_column is None:
            values_column = df.columns[0]
        fig = df.plot(kind='bar', figsize=(20, 10), title=values_column + ' per Era', legend=False, xticks=[],
                      snap=False).get_figure()
        return fig
