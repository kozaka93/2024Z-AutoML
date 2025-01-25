import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SingleStockDataPreprocessor:
    """
    A class for preprocessing stock market data, including feature engineering,
    scaling, and missing value handling.

    Attributes:
        data (pd.DataFrame): Input data with a DatetimeIndex.
        price_columns (list): List of columns containing price data.
        volume_column (str): Column containing volume data.
        original_data (pd.DataFrame): Copy of the original input data.
        prices (list): List of mean prices computed from price_columns.
        preprocessed (bool): Flag indicating whether data has been preprocessed.
    """

    def __init__(self, data, price_columns, volume_column):
        """
        Initializes the preprocessor with stock data and relevant column names.

        Args:
            data (pd.DataFrame): Input data with a DatetimeIndex.
            price_columns (list): List of columns with price data.
            volume_column (str): Column with volume data.

        Raises:
            ValueError: If the input data does not have a DatetimeIndex.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("The input data must have a DatetimeIndex.")
        
        self.volume_column = volume_column
        self.price_columns = price_columns
        self.data = data.copy()
        self.original_data = data.copy()
        self.data['price'] = self.data[price_columns].mean(axis=1)
        self.prices = self.data['price'].tolist()
        self.preprocessed = False

    def fill_missing_values(self):
        """
        Fills missing values in the dataset using linear interpolation.
        """
        self.data.interpolate(method='linear', inplace=True)

        # If first or last value in column is missing, fill with next/previous value    
        self.data.fillna(method='ffill',inplace=True)
        self.data.fillna(method='bfill',inplace=True)
        self.original_data = self.data.copy()
        self.prices = self.data['price'].tolist()

    def calculate_rsi(self, window):
        """
        Calculates the Relative Strength Index (RSI) for the given window size.

        Args:
            window (int): Look-back window for RSI calculation.
        """
        delta = self.data['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

    def calculate_technical_indicators(self):
        """
        Computes various technical indicators, including percentage change,
        moving averages, RSI, and volatility.
        """
        ma_1 = 10  # Moving average window 1
        ma_2 = 50  # Moving average window 2
        rsi = 50   # RSI window

        # Percentage change
        self.data['pct_change'] = self.data['price'].pct_change().fillna(0)

        # Price range
        self.data['range'] = self.data[self.price_columns].max(axis=1) - self.data[self.price_columns].min(axis=1)

        # Moving averages
        self.data[f'MA_{ma_2}'] = self.data['price'].rolling(window=ma_2).mean()
        self.data[f'MA_{ma_1}'] = self.data['price'].rolling(window=ma_1).mean()

        # Fill initial moving average values
        for i in range(ma_2):
            self.data[f'MA_{ma_2}'].iloc[i] = np.mean(self.prices[:i + 1])
        for i in range(ma_1):
            self.data[f'MA_{ma_1}'].iloc[i] = np.mean(self.prices[:i + 1])

        # Moving average difference and indicator
        self.data['MA_diff'] = (self.data[f'MA_{ma_1}'] - self.data[f'MA_{ma_2}']) / self.data[f'MA_{ma_2}']
        self.data['MA_indicator'] = pd.cut(
            self.data['MA_diff'],
            bins=[-float('inf'), 0, float('inf')],
            labels=[0, 1],
            right=False
        ).astype(int)

        # Volatility
        self.data['Volatility'] = self.data['pct_change'].rolling(window=20).std()
        for i in range(5, 20):
            self.data['Volatility'].iloc[i] = np.std(self.data['pct_change'][:i + 1])
        for i in range(5):
            self.data['Volatility'].iloc[i] = np.std(self.data['pct_change'][:5])

        # RSI
        self.calculate_rsi(window=rsi)
        first_nonnull_value = self.data['RSI'].dropna().iloc[0] if not self.data['RSI'].dropna().empty else None
        if first_nonnull_value is not None:
            self.data['RSI'].fillna(first_nonnull_value, inplace=True)

        self.data['RSI_level'] = pd.cut(
            self.data['RSI'],
            bins=[-float('inf'), 50, float('inf')],
            labels=[0, 1],
            right=False
        ).astype(int)
        self.data['RSI_change'] = self.data['RSI'].pct_change()
        first_nonnull_value = self.data['RSI_change'].dropna().iloc[0] if not self.data['RSI_change'].dropna().empty else None
        if first_nonnull_value is not None:
            self.data['RSI_change'].fillna(first_nonnull_value, inplace=True)


    def scale_columns(self, columns):
        """
        Scales specified columns using MinMaxScaler.

        Args:
            columns (list): List of column names to scale.
        """
        scaler = MinMaxScaler()
        for column in columns:
            self.data[column] = scaler.fit_transform(self.data[[column]])

    def preprocess(self):
        """
        Executes the full preprocessing pipeline, including filling missing values,
        calculating technical indicators, and scaling selected columns.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        if not self.preprocessed:
            self.fill_missing_values()
            self.calculate_technical_indicators()
            self.scale_columns(columns=self.data.columns.tolist())
            self.preprocessed = True

        return self.data

    def make_np_array(self):
        """
        Converts the preprocessed data to a NumPy array.

        Returns:
            np.ndarray: Data as a NumPy array.
        """
        return self.data.to_numpy()
