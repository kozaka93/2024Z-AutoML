import pandas as pd
import numpy as np
from numerai_automl.visual.radar_plot import RadarPlot
import matplotlib.pyplot as plt


def test_radar_plot():
    df2 = pd.DataFrame({
        # random number from 0 to 1
        'mean': np.random.rand(5),
        'std': np.random.rand(5),
        'sharpe': np.random.rand(5),
        'max_drawdown': np.random.rand(5),
    })
    df2.index = ['ensemble_' + str(i) for i in range(5)]
    print("DATA:")
    print(df2)
    print("RADAR PLOT:")
    rp = RadarPlot(df2)
    fig = rp.get_plot()
    plt.show()


if __name__ == '__main__':
    test_radar_plot()
