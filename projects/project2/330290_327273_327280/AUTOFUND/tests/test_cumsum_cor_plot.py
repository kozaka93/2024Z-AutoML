from numerai_automl.scorer.scorer import Scorer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numerai_automl.visual.cumsum_cor_features_plot import CumsumCorrFeaturesPlot
import matplotlib.pyplot as plt
import seaborn as sns

def test_cumsum_cor_plot():

    scorer = Scorer()

    # stwórz dataframe z danymi testowymi
    # era to nie będą liczby od 1 do 500 po 10 razy każda

    df = pd.DataFrame({
        'era': np.repeat(np.arange(1, 501), 10),
        # losowe wartości z [0,0.2, 0.5,0.8,1]
        'target': np.random.choice([0,0.2, 0.5,0.8,1], 5000)
    })
    df["prediction_m1"] = df['target'] + np.random.normal(0, 0.1, 5000)
    df["prediction_m2"] = df['target'] ** 2 + np.random.normal(0, 0.1, 5000)
    df["prediction_m3"] = df['target'] ** 2
    df["prediction_m4"] = np.random.choice([0,0.2, 0.5,0.8], 5000)
    df['main_model'] = np.random.choice([0,0.2, 0.5,0.8,1], 5000)
    #minmax scaling for predictions
    minmax=MinMaxScaler()
    df[['prediction_m1','prediction_m2','prediction_m3']]=minmax.fit_transform(df[['prediction_m1','prediction_m2','prediction_m3']])
    print("DATA input:")
    print(df)
    print("DATA before plotting:")
    df2=scorer.compute_cumsum_correlation_per_era(df,'target')
    print(df2)
    # Usuwamy prefiks "prediction_" z nazw kolumn
    pr = CumsumCorrFeaturesPlot(df2)
    fig = pr.get_plot()
    fig.show()

    plt.savefig('cumsum_cor_plot.png')
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    test_cumsum_cor_plot()