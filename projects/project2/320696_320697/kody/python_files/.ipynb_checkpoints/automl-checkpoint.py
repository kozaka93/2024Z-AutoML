#------------------------------------------IMPORT LIBS-------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from skmultilearn.model_selection import iterative_train_test_split

#------------------------------------------IMPORT LIBS-------------------------------------------


#------------------------------------------PREPROCESS AND EVALUTAION-------------------------------------------
from python_files.Preprocess import PaddingEstimator, add_pad, extract_features_with_window, process_labels_with_window, WindowFeatureExtractor, WindowLabelProcessor, process_labels_with_window_2d, PCADimensionReducer
from python_files.Plot import compute_and_plot_statistics, plot_statistics_per_class, plot_data_raport
#------------------------------------------PREPROCESS AND EVALUTAION-------------------------------------------



# -----------------------------IMPORT piplines and param distributions-----------------------#
from python_files.ModelSelection import display_model_info, param_distributions, pipelines
# -----------------------------IMPORT piplines and param distributions-----------------------#


class AutoMlMultiLabelClassifier:
    def __init__(self, model=None, window_size=120, step_size=120, labels_type=None):
        """

        Args:

        """
        
        self.model = None
        self.is_fitted = False
        self.window_size = window_size # Rozmiar okna dla obrabiania X
        self.step_size = step_size # Rozmiar okna dla obrabiania y
        self.labels_type = labels_type 
        
    def trimming_data(self, X, y, fraction = 0.5):
        # Trimming z wykorzstyaniem funkcji pojedyńczych
        # print("X.shape: ",X.shape)
        X = self.trimming_data_X(X, fraction)
        # print("X.shape: ",X.shape)
        y = self.trimming_data_y(y, fraction)

        # print("y.shape: ",y.shape)
        return X, y

    def trimming_data_y(self, y, fraction = 0.5):
        # Znajdź minimalną długość
        min_length = min(arr.shape[0] for arr in y)
        self.window_size = int(min_length*fraction)
        self.step_size =  int(min_length*fraction)
        # print("otrzymane okno:", self.window_size )
        min_length = self.window_size

        trimmed_segments_y = [
            arr[i:i+min_length, :] 
            for arr in y 
            for i in range(0, arr.shape[0] - min_length + 1, min_length)
        ]
        
        y = np.array(trimmed_segments_y)

        y = y.reshape(-1, y.shape[2])
        y = process_labels_with_window_2d(y, self.window_size, self.step_size)

        # # y = y.reshape(y.shape[0], -1) 
        # y = pd.DataFrame(y)

        # y.columns = ['label_' + str(col) for col in y.columns]

        return y

    def trimming_data_X(self, X, fraction = 0.5):
        min_length = min(arr.shape[0] for arr in X)
        self.window_size =  int(min_length*fraction)
        self.step_size =  int(min_length*fraction)
        # print("otrzymane okno:", self.window_size )
        min_length = self.window_size
        # Wyodrębnienie maksymalnej liczby segmentów o długości min_length
        trimmed_segments = [
            arr[i:i+min_length, :] 
            for arr in X 
            for i in range(0, arr.shape[0] - min_length + 1, min_length)
        ]
        
        X = np.array(trimmed_segments)
        # self.print("X.shape: ",X.shape)

        X = X.reshape(X.shape[0], -1) 
        # X = X.reshape(-1,X.shape[2]) 
        
        # print("X.shape: ",X.shape)
        # X = pd.DataFrame(X)
    
        # X.columns = ['feature_' + str(col) for col in X.columns]

        return X
    
    def fit(self, X, y):
        """
        Funkcja dokonuje selekcji i optymalizacji mogelu. Piplines i paramdistributions są dostępne w ModelSelection.py
        Wybiera spośród:
        - SVC OneVsRest
        - Regresja logistyczna OneVsRest
        - XGBoost OneVsRest
        - XGBoost Mulitoutput
    
        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Etykiety (labels).
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)

            best_model = None
            best_score = 0
            best_params = {}
    
            model_results = []  # Lista do przechowywania wyników modeli
    
            for name, pipeline in pipelines.items():
                print(f"---------------------------------------------------")
                print(f"Training model: {name}")
    
                grid_search = GridSearchCV(pipeline, param_distributions[name]
                                           , scoring='f1_macro', n_jobs=4, error_score='raise')
                grid_search.fit(X_train, y_train)
    
                print(f"{name} - Best Parameters: {grid_search.best_params_}")
                print(f"{name} - Best Cross-Validation Score: {grid_search.best_score_}")
    
                model_results.append((name, grid_search.best_score_, grid_search.best_params_))  # Dodanie wyników modelu
    
                if grid_search.best_score_ > best_score:
                    best_model = grid_search.best_estimator_
                    best_score = grid_search.best_score_
                    best_params = grid_search.best_params_
                print(f"---------------------------------------------------")

    
            model_results.sort(key=lambda x: x[1], reverse=True)
    
            print("\nModels ranking:")
            for idx, (name, score, params) in enumerate(model_results):
                print(f"{idx + 1}. {name} - Best CV Score: {score} - Best Parameters: {params}")
    
            # Testowanie najlepszego modelu
            y_pred = best_model.predict(X_test)
    
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
    
            self.is_fitted = True
            self.model = best_model
    
            display_model_info(self.model)
            print("Test Set statistics")
            print(f"Accuracy: {accuracy}")
            print(f"Recall: {recall}")
            print(f"Precision: {precision}")
    
        except Exception as e:
            print(f"Wystąpił błąd podczas treningu: {e}")
            
    def predict(self, X):
        """
        Przewiduje etykiety dla podanych danych.

        Args:
            X (np.ndarray): Dane wejściowe (features).

        Returns:
            np.ndarray: Przewidywane etykiety.
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj metody fit przed predict.")
        try:
            predictions = self.model.predict(X)
            
            return predictions
        except Exception as e:
            print(f"Błąd podczas przewidywania: {e}")
            return None

    def score(self, X, y, plot_stats=False):
        """
        Oblicza dokładność modelu na podanych danych testowych.

        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Rzeczywiste etykiety (labels).

        Returns:
            dictinary: statistics - słownik zawierający informację o wszystkich najważniejszych statystykach, dla każdej klasy.

       statistics = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'auc': auc_scores,
            'accuracy': accuracy_scores
        }
        """

        
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj metody fit przed score.")
        try:
            y_pred = self.predict(X)
            
            statistics = compute_and_plot_statistics(np.array(y), y_pred, self.labels_type, print_stats=False, plot_stats=plot_stats)
            
            return statistics['accuracy']
        except Exception as e:
            print(f"Błąd podczas obliczania dokładności: {e}")
            return None

    def raport_scores(self, X, y):
        """
        Funkcja wyświetla najważniejsze statystki dla modelu. Tworzy wyrkesu i podsumowania. Szczególnie liczy efektywnośc modelu dla każdej klasy oddzielnie.

        Args:
            dictionary: statistics - słownik zawierający informację o wszystkich najważniejszych statystykach, dla każdej klasy.

       statistics = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'auc': auc_scores,
            'accuracy': accuracy_scores
        }
        """
        print("Wykres przedstawiający statystyki (precision, recall, f1, AUC oraz accuracy) dla każdej klasy błędów z osobna")
        # Wykresy dla statystyk dla różnych klas.
        y_pred = self.predict(X)
        plot_stats = False   
        n_features = y.shape[1]
        gen_labels = [f'label_{i}' for i in range(n_features)]

        statistics = compute_and_plot_statistics(np.array(y), y_pred,        gen_labels if not self.labels_type else self.labels_type, print_stats=False, plot_stats=plot_stats)
        plot_statistics_per_class(statistics,      gen_labels if not self.labels_type else self.labels_type)

        # Poniżej wypisujemy wyniki w postaci liczbowej
        statistics_list = sorted(statistics)  # Sortujemy metryki alfabetycznie
        statistics_dict = {metric: statistics[metric] for metric in statistics_list}
        
        # Tworzenie DataFrame
        df = pd.DataFrame.from_dict(statistics_dict, orient="index")
        df.columns = [f"Value {gen_labels[i] if not self.labels_type else self.labels_type[i]}" for i in range(df.shape[1])]  # Nazwy kolumn
        
        # Wyświetlenie macierzy
        print("Wyniki w postaci macierzy")
        print(df)


    def raport_data(self, X, y):
        """
        Funkcja wyświetla najważniejsze statystki i informacje na temat zestawu danych.

        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Rzeczywiste etykiety (labels).

        """

        n_features = y.shape[1]
        gen_labels = [f'label_{i}' for i in range(n_features)]
        plot_data_raport(X, y,  gen_labels if not self.labels_type else self.labels_type)
        


