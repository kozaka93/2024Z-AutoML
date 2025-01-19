import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

class Imputer:
    def __init__(self, target_column, linear_correlation_threshold=0.8, rf_correlation_threshold=0.2):
        """
        Initialize the imputer.

        Parameters:
        - linear_correlation_threshold (float): Threshold above which linear regression will be used.
        - rf_correlation_threshold (float): Threshold for moderate correlation to use Random Forest.
        """
        self.linear_correlation_threshold = linear_correlation_threshold
        self.rf_correlation_threshold = rf_correlation_threshold
        self.imputation_info = {}  # Dictionary to store imputation method info for each column
        self.target_column = target_column

    def impute_missing_values(self, dataframe):
        """
        Impute missing values based on correlation with the target column.

        Parameters:
        - dataframe (pd.DataFrame): The dataframe to process.
        - self.target_column (str): The name of the target column.

        Returns:
        - pd.DataFrame: DataFrame with missing values imputed.
        """
        df_copy = dataframe.copy()

        # Check if target column is categorical, if yes, encode it
        if df_copy[self.target_column].dtype == 'object' or df_copy[self.target_column].dtype.name == 'category':
            le_target = LabelEncoder()
            df_copy[self.target_column] = le_target.fit_transform(df_copy[self.target_column])
            target_is_categorical = True
        else:
            target_is_categorical = False
        
        # Filter only numerical columns for correlation calculation
        numerical_columns = df_copy.select_dtypes(include=['number']).columns

        # Compute correlations only for numerical columns
        correlations = df_copy[numerical_columns].corr()[self.target_column]

        for column in df_copy.columns:
            if column == self.target_column:
                continue
            if df_copy[column].isnull().any():  # Check if the column has missing values
                # For numerical columns
                if column in numerical_columns:
                    correlation = abs(correlations.get(column, 0))  # Get correlation, default to 0 if not found

                    # If the correlation is strong (linear relationship), use linear regression
                    if correlation >= self.linear_correlation_threshold:
                        X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)
                        y = df_copy.dropna(subset=[column])[column].values

                        model = LinearRegression()
                        model.fit(X, y)

                        missing_values = df_copy[df_copy[column].isnull()]
                        predicted_values = model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                        df_copy.loc[df_copy[column].isnull(), column] = predicted_values

                        # Store imputation method for this column
                        self.imputation_info[column] = {
                            "Imputation Method": "Linear Regression",
                            "Correlation": correlation,
                        }

                    # If the correlation is moderate, use random forest regressor
                    elif correlation >= self.rf_correlation_threshold:
                        X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)
                        y = df_copy.dropna(subset=[column])[column].values

                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X, y)

                        missing_values = df_copy[df_copy[column].isnull()]
                        predicted_values = rf_model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                        df_copy.loc[df_copy[column].isnull(), column] = predicted_values

                        # Store imputation method for this column
                        self.imputation_info[column] = {
                            "Imputation Method": "Random Forest",
                            "Correlation": correlation,
                        }

                    # If the correlation is weak, use median for imputation
                    else:
                        df_copy[column] = df_copy[column].fillna(df_copy[column].median())

                        # Store imputation method for this column
                        self.imputation_info[column] = {
                            "Imputation Method": "Median",
                            "Correlation": correlation,
                        }

                # For categorical columns
                else:
                    # Encode categorical column
                    le = LabelEncoder()
                    non_null_data = df_copy.loc[df_copy[column].notnull(), column]
                    le.fit(non_null_data)
                    df_copy.loc[df_copy[column].notnull(), column] = le.transform(non_null_data)

                    # Preprocessing data for training
                    X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)
                    y = df_copy.dropna(subset=[column])[column].astype(int).values  # Ensure y is int for categorical column

                    # Training the model
                    model = DecisionTreeClassifier(random_state=42)
                    model.fit(X, y)

                    # Predicting missing values
                    missing_values = df_copy[df_copy[column].isnull()]
                    predicted_values = model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                    # Filling missing values in the DataFrame
                    df_copy.loc[df_copy[column].isnull(), column] = predicted_values
                    df_copy[column] = le.inverse_transform(df_copy[column].astype(int))

                    # Store imputation method for this column
                    self.imputation_info[column] = {
                        "Imputation Method": "Decision Tree",
                        "Correlation": None,  # Correlation doesn't apply to categorical columns
                    }

        # If the target column was categorical, decode it back to its original form
        if target_is_categorical:
            df_copy[self.target_column] = le_target.inverse_transform(df_copy[self.target_column])

        return df_copy

    def get_imputation_info(self):
        """
        Return the imputation method used for each column.

        Returns:
        - dict: A dictionary with column names as keys and imputation methods as values.
        """
        return self.imputation_info
