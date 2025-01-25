# TODO: CHECK IF THE SCORER IS WORKING CORRECTLY
from numerai_automl.scorer.scorer import Scorer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def test_mmc():
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
    x,y,z = scorer.get_mmc(df, meta_model_col='main_model', target='target')
    print("PER ERA MMC:")
    print(x)
    print("CUMSUM MMC:")
    print(y)
    print("SUMMARY:")
    print(z)


if __name__ == '__main__':
    test_mmc()