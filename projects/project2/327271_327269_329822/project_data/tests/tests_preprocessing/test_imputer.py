import pandas as pd
from medaid.preprocessing.imputer import Imputer

def test_imputation():

    data_path = "data/binary/cardio_train.csv" 
    df = pd.read_csv(data_path, sep=";")

    target_column = "cardio"

    print("Original DataFrame with missing values:")
    print(df)

    
    imputer = Imputer() 
    df_imputed = imputer.impute_missing_values(df, target_column=target_column)


    print("\nDataFrame after imputation:")
    print(df_imputed)


    assert df_imputed.isnull().sum().sum() == 0, "There are still missing values in the DataFrame!"

    print("\nTest passed: No missing values remaining.")


test_imputation()
