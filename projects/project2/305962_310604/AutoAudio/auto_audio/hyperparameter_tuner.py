from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from .models.base_model import AutoAudioBaseModel


class HyperparameterTuner:
    def __init__(
        self,
        search_method="random",
        scoring="accuracy",
        cv=5,
        random_state=42,
        n_iter=30,
    ):
        """
        Klasa pozwalająca na optymalizację hiperparametrów modelu.

        Parameters:
        - search_method: Metoda optymalizacji ("random", "bayes").
        - scoring: Metryka oceny modelu (np. "accuracy", "f1", etc.).
        - cv: Liczba podziałów w walidacji krzyżowej.
        - random_state: Losowy seed dla powtarzalności.
        """
        self.search_method = search_method
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_iter = n_iter

    def tune(self, model: AutoAudioBaseModel, X, y) -> BaseEstimator:
        """
        Optymalizuje hiperparametry modelu.

        Parameters:
        - model: Model do optymalizacji.
        - X: Cechy treningowe.
        - y: Etykiety treningowe.
        - n_iter: Liczba kombinacji do sprawdzenia.

        Returns:
        - Najlepszy model z wybranymi hiperparametrami.
        """

        param_ranges = model.get_param_ranges(self.search_method)
        if not param_ranges:
            raise ValueError(
                f"Model {model} does not provide parameter ranges for {self.search_method} search."
            )

        if self.search_method == "random":
            search = RandomizedSearchCV(
                estimator=model.get_model(),
                param_distributions=param_ranges,
                scoring=self.scoring,
                n_iter=self.n_iter,
                cv=self.cv,
                random_state=self.random_state,
                verbose=1,
                n_jobs=-1,
            )
        elif self.search_method == "bayes":
            search = BayesSearchCV(
                estimator=model.get_model(),
                search_spaces=param_ranges,
                scoring=self.scoring,
                n_iter=self.n_iter,
                cv=self.cv,
                random_state=self.random_state,
                verbose=0,
                n_jobs=-1,
            )
        else:
            raise ValueError("Invalid search method. Choose 'random', or 'bayes'.")

        search.fit(X, y)
        print(f"Best parameters found: {search.best_params_}")
        return search.best_estimator_

