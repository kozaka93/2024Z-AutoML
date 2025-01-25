from numerai_automl.visual.per_era_bar_plot import PerEraBarPlot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def test_per_era_bar_plot():
    df = pd.DataFrame({
        'era': np.repeat(np.arange(1, 501), 10),
        'values': np.random.choice([-0.8, -0.5, 0, 0.2, 0.5, 0.8, 1], 5000)
    })
    df = df.groupby('era').mean()
    # print("DATA:")
    # print(df)
    # print("BAR PLOT:")

    # df = pd.read_csv("return_data.csv")
    rp = PerEraBarPlot(df)
    fig = rp.get_plot()
    plt.show()

    plt.savefig('per_era_bar_plot.png')
    plt.close()  # Close the figure to free memory


if __name__ == '__main__':
    test_per_era_bar_plot()
