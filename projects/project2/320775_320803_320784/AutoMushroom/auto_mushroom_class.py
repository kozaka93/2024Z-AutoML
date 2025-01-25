from sklearn.model_selection import train_test_split
import time

from AutoMushroom.preprocessing import prep
from AutoMushroom.model_selection import *
from AutoMushroom.report import *

import warnings


class AutoMushroom:
    warnings.filterwarnings("ignore", category=FutureWarning)

    def __init__(self):

        self.__selected_features = None
        self.__X = None
        self.__y = None
        self.__fit_time = None
        self.__voting = None
        self.best_model = None
        self.best_score = None
        self.metrics = None

    def fit(self, X, y, mode = 'medium', voting = 'hard'):
        TEST_SIZE = 0.1
        RANDOM_STATE = 10

        self.__voting = voting

        try:
            start_time = time.time()

            self.__X = X
            self.__y = y

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

            # preprocesing
            X_train_preprocessed, self.__selected_features = prep(X_train, y_train, mode='train')
            X_test_preprocessed = prep(X_test, mode='test', features=self.__selected_features)

            # model selection
            self.best_model, self.best_score = model_selection(X_train_preprocessed, y_train, mode = mode, voting = voting)
            if self.best_model is None:
                raise ValueError("Model selection failed. No model was returned.")

            # evaluate on test set
            y_pred = self.best_model.predict(X_test_preprocessed)
            self.metrics = model_evaluation(y_test, y_pred)

            self.__fit_time = time.time() - start_time


        except Exception as e:
            print(f"An error occurred during the AutoML process: {str(e)}")
            raise

    def predict(self, X):

        if not self.best_model:
            raise Exception("Model is not trained yet. Call fit() before predict().")

        if self.__voting == 'soft' and isinstance(self.best_model, VotingClassifier):
            raise Exception("Hard predictions are not supported with soft voting.")

        # preprocess the input data
        X_preprocessed = prep(X, mode='test', features=self.__selected_features)

        # predict
        return self.best_model.predict(X_preprocessed)

    def predict_proba(self, X):

            if not self.best_model:
                raise Exception("Model is not trained yet. Call fit() before predict().")

            if self.__voting == 'hard' and isinstance(self.best_model, VotingClassifier):
                raise Exception("Predicting probabilities is not supported with hard voting.")

            # preprocess the input data
            X_preprocessed = prep(X, mode='test', features=self.__selected_features)

            # predict
            return self.best_model.predict_proba(X_preprocessed)


    def summary_report(self):

        if not self.best_model:
            raise Exception("Model is not trained yet. Call fit() before summary_report().")

        print("Pakiet AutoMushroom dla grzybiarzy")
        print("Analizowane są zbiory danych z podziałem na klasy 0 lub 1, gdzie 0 oznacza jadalny grzyb, a 1 trujący.")
        print("Analiza danych:")
        data_overview(self.__X)
        print("Balans klas:")
        plot_mushroom_balance(self.__y)
        print("Preprocessing składa się z kilku etapów:")
        print("Numeryczne dane są wypełniane średnią w przypadku braków, a następnie skalowane do zakresu [0,1] przy użyciu MinMaxScaler.")
        print("Dane kategoryczne są uzupełniane najczęściej występującymi wartościami, a następnie kodowane za pomocą metody one-hot encoding.")
        print("W trybie treningowym wybierane są istotne cechy za pomocą klasyfikatora Random Forest i SelectFromModel, a dane testowe są ograniczane do wybranych cech.")
        print("Ważność cech:")
        summarize_selected_features(self.__selected_features)

        print("Analiza jakości modeli i konfiguracja finalnego komitetu:")
        print("1. Miara oceny modeli:")
        print("   Do analizy jakości modeli wykorzystano kombinację ważonych miar ROC AUC oraz Recall:")
        print("   Custom Score = (Recall: 0.3, ROC AUC: 0.7)")
        print()
        print(
            "2. Modele użyte w analizie: KNeighborsClassifier, GradientBoostingClassifier, RandomForestClassifier, LogisticRegression")
        print(
            "   Dodatkowo komitet VotingClassifier z wyżej wymienionych modeli z optymalnymi parametrami")
        print()
        print("3. Optymalizacja parametrów:")
        print(
            "   Dla każdego z modeli, przy użyciu metody RandomizedSearch, dobrano najlepsze zestawy hiperparametrów.")
        print()
        print("4. Parametry finalnego modelu:")
        print(self.best_model)
        print()
        print(f"5. Czas trenowania modelu: {self.__fit_time} seconds")
        print()
        print("6. Wynik Custom Score:")
        print(f"   Uzyskana wartość Custom Score dla tego modelu na zbiorze walidacyjnym wynosiła: {self.best_score}")
        # Plot Confusion Matrix
        plot_confusion_matrix(self.metrics['confusion_matrix'])
        # Plot ROC AUC Curve
        plot_roc_auc_curve(self.metrics['roc_curve'][0], self.metrics['roc_curve'][1], self.metrics['roc_auc'])
        # Plot Bar Plot of Metrics
        plot_metrics_bar(self.metrics, self.best_score)

        generate_model_analysis_from_metrics(self.metrics)

