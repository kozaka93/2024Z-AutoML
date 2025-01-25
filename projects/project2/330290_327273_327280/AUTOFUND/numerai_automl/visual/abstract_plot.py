import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod


class AbstractPlot(ABC):
    """
    Abstract class for all classes that will generate plots based on data given.
    """

    def __init__(self, data_for_visualization: pd.DataFrame):
        """
        Constructor for AbstractPlot class.
        :param data_for_visualization: The data that will be used to generate the plot.
        """
        self.data_for_visualization = data_for_visualization

    @abstractmethod
    def get_plot(self) -> plt.Figure:
        """
        Abstract method that will generate the plot using the data given in constructor.
        :return plt.Figure: The plot that was generated.
        """
        pass
