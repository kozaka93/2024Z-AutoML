import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.fft import fft
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import TSNE
#-------------------------------------------------------------------------------------------------
#                                Loading
#-------------------------------------------------------------------------------------------------

def load_pickle_file(filename):
    """
    Wczytuje dane z pliku .pickle.
    
    Args:
        filename (str): Ścieżka do pliku .pickle.
    
    Returns:
        dict: Słownik z danymi wczytanymi z pliku.
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Dane wczytane z pliku '{filename}'.")
        return data
    except FileNotFoundError:
        print(f"Błąd: Plik '{filename}' nie został znaleziony.")
        return None
    except pickle.UnpicklingError:
        print(f"Błąd: Nie udało się wczytać danych z pliku '{filename}'.")
        return None

from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
import numpy as np

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)


one_rep = 50
divison = 1

def extract_features_with_window(X, window_size=one_rep, step_size=(one_rep//divison)):
    features = []
    
    for seq in X:
        # Przesuwanie okna po sekwencji
        for start_idx in range(0, seq.shape[0] - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = seq[start_idx:end_idx]
            
            window_features = []
            for sample in window:
                # Statystyki opisowe
                # if np.all(sample == 0):
                #     continue  # Pomijamy próbkę, jeśli jest pełna zer
                mean = np.mean(sample)
                std = np.std(sample)
                min_val = np.min(sample)
                max_val = np.max(sample)
                median = np.median(sample)
                
                # Różnice pierwszego rzędu
                diff = np.diff(sample)
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)

                # FFT (transformacja Fouriera)
                fft_values = np.abs(fft(sample))
                fft_mean = np.mean(fft_values)
                fft_std = np.std(fft_values)

                # Dodajemy cechy do okna
                window_features.extend([mean, std, min_val, max_val, median, mean_diff, std_diff, fft_mean, fft_std])
            
            features.append(window_features)
    
    print('pracuje 3')
    print(type(features))
    print('len(features): ', len(features))
    return np.array(features)
                

def process_labels_with_window(y, window_size=one_rep, step=(one_rep // divison)):
    window_labels = []
    
    for seq in y:
        # Przesuwanie okna po sekwencji
        for start in range(0, seq.shape[0] - window_size + 1, step):
            end = start + window_size
            window = seq[start:end]  # Wyciągamy okno
            
            # Dla każdej klasy (kolumny) w oknie, jeśli występuje co najmniej 5 '1', ustawiamy etykietę tej klasy na '1'
            window_label = (np.sum(window == 1, axis=0) >= 5).astype(int)  # Zwracamy wektor 10-elementowy
            window_labels.append(window_label)
    
    return np.array(window_labels)



def add_pad(X, y=None):
    # Upewnij się, że X jest listą sekwencji
    # [seq.reshape(seq.shape[0], -1).shape for seq in X]
        
    max_len = max([seq.shape[0] for seq in X])  # Maksymalna długość sekwencji
    X = pad_sequences([seq.reshape(seq.shape[0], -1) for seq in X], maxlen=max_len, padding='post', dtype='float32').reshape(len(X), max_len, 136)
    y = pad_sequences(y, maxlen=max_len, padding='post', dtype='int')
                
    return X, y


class PaddingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Możesz dodać dodatkowe parametry, jeśli chcesz
    
    def fit(self, X, y=None):
        # Metoda fit nie robi nic, bo nie uczymy się niczego z danych
        return self
    
    def transform(self, X, y=None):
        # Upewnij się, że X jest listą sekwencji
        # [seq.reshape(seq.shape[0], -1).shape for seq in X]
        
        max_len = max([seq.shape[0] for seq in X])  # Maksymalna długość sekwencji
        X = pad_sequences([seq.reshape(seq.shape[0], -1) for seq in X], maxlen=max_len, padding='post', dtype='float32').reshape(len(X), max_len, 136)
        y = pad_sequences(y, maxlen=max_len, padding='post', dtype='int')
                
        return X, y



class PCADimensionReducer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=80):
        self.n_components = n_components
        self.pca = None

    def fit(self, X, y=None):
        # Ustalamy liczbę komponentów na podstawie liczby cech w danych
        n_features = X.shape[1]
        self.n_components = min(self.n_components, n_features)
        
        # Dopasowujemy PCA do danych
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        return self

    def transform(self, X, y=None):
        # Transformujemy dane przy użyciu dopasowanego PCA
        return self.pca.transform(X)





class WindowFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def fit(self, X, y=None):
        # Nie wymaga uczenia, więc po prostu zwracamy self
        return self

    def transform(self, X, y=None):
        # Sprawdzamy, czy X to DataFrame
        if isinstance(X, pd.DataFrame):
            # Konwertujemy DataFrame na numpy array
            X = X.values
        
        features = []

        # Iterujemy po próbkach w danych
        for start_idx in range(0, X.shape[0] - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window = X[start_idx:end_idx]
            
            window_features = []
            for sample in window.T:  # Transponujemy, aby iterować po kolumnach
                # Statystyki opisowe
                mean = np.mean(sample)
                std = np.std(sample)
                min_val = np.min(sample)
                max_val = np.max(sample)
                median = np.median(sample)
                
                # Różnice pierwszego rzędu
                diff = np.diff(sample)
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)
    
                # FFT (transformacja Fouriera)
                fft_values = np.abs(fft(sample))
                fft_mean = np.mean(fft_values)
                fft_std = np.std(fft_values)
    
                window_features.extend([mean, std, min_val, max_val, median, mean_diff, std_diff, fft_mean, fft_std])
            
            features.append(window_features)
        
        return pd.DataFrame(features)



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WindowLabelProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step):
        self.window_size = window_size
        self.step = step

    def fit(self, X, y=None):
        # Nie wymaga uczenia, więc po prostu zwracamy self
        return self

    def transform(self, y):
        # Sprawdzamy, czy y to DataFrame
        if isinstance(y, pd.DataFrame):
            # Konwertujemy DataFrame na numpy array
            y = y.values
        
        window_labels = []
        
        for start in range(0, y.shape[0] - self.window_size + 1, self.step):
            end = start + self.window_size
            window = y[start:end]  # Wyciągamy okno
            
            window_label = (np.sum(window == 1, axis=0) >= 5).astype(int)  # Zwracamy wektor 10-elementowy
            window_labels.append(window_label)
        
        return np.array(window_labels)




def process_labels_with_window_2d(y, window_size, step):
    window_labels = []
    
    # Iterujemy po próbkach w danych
    for start in range(0, y.shape[0] - window_size + 1, step):
        end = start + window_size
        window = y[start:end]  # Wyciągamy okno
        
        # Dla każdej klasy (kolumny) w oknie, jeśli występuje co najmniej 5 '1', ustawiamy etykietę tej klasy na '1'
        window_label = (np.sum(window == 1, axis=0) >= int(window_size/5)).astype(int)  # Zwracamy wektor 10-elementowy
        window_labels.append(window_label)
    
    return np.array(window_labels)







