import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class ModelSelector:
    """
    Klasa odpowiedzialna za automatyczny wybór i strojenie modeli klasyfikacyjnych
    przy użyciu GridSearchCV lub RandomizedSearchCV.
    """

    def __init__(self, search_method="grid", scoring="accuracy", n_iter=10, cv=3, random_state=42):
        """
        :param search_method: 'grid' lub 'random'
        :param scoring: metryka (np. accuracy, f1, roc_auc)
        :param n_iter: liczba losowań w Random Search
        :param cv: liczba foldów w cross validation
        :param random_state: ustalenie losowości
        """
        valid_search_methods = ["grid", "random"]
        valid_scoring = ["accuracy", "f1", "roc_auc"]

        assert search_method in valid_search_methods, f"Invalid search_method '{search_method}'. Must be one of {valid_search_methods}."
        assert scoring in valid_scoring, f"Invalid scoring '{scoring}'. Must be one of {valid_scoring}."

        self.search_method = search_method
        self.scoring = scoring
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.best_model = None
        self.scores_history = []

    def fit(self, X, y):
        """
        Przeszukuje kilka wybranych modeli i wybiera ten, który daje najlepszy wynik (best_score).
        """
        candidates = [
            {
                "model": DecisionTreeClassifier(random_state=self.random_state, class_weight='balanced'),
                "params": {
                    "model__max_depth": [None, 5, 10, 20, 50],
                    "model__min_samples_split": [2, 5, 10, 20, 50],
                    "model__min_samples_leaf": [1, 2, 5, 10, 20]
                }
            },
            {
                "model": RandomForestClassifier(random_state=self.random_state, class_weight='balanced'),
                "params": {
                    "model__n_estimators": [100, 300, 500, 700, 1000],
                    "model__max_depth": [None, 10, 20, 30, 50],
                    "model__min_samples_split": [2, 5, 10, 20],
                    "model__min_samples_leaf": [1, 2, 5, 10],
                    "model__max_features": ["sqrt", "log2", None],
                    "model__bootstrap": [True, False]
                }
            },
            {
                "model": SVC(random_state=self.random_state, probability=True, class_weight='balanced'),
                "params": {
                    "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "model__gamma": ["scale", "auto"],
                    "model__degree": [2, 3, 4, 5],
                }
            },
            {
                "model": XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss'),
                "params": {
                    "model__n_estimators": [100, 300, 500, 700],
                    "model__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
                    "model__max_depth": [3, 5, 7, 10, 15],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                    "model__reg_alpha": [0, 0.1, 0.5, 1, 2],
                    "model__reg_lambda": [1, 2, 3, 5, 10]
                }
            }
        ]


        best_score = -np.inf
        best_estimator = None

        for candidate in candidates:
            print(f"Training model: {candidate['model'].__class__.__name__}")
            pipe = Pipeline([("model", candidate["model"])])

            if self.search_method == "grid":
                search = GridSearchCV(
                    pipe,
                    param_grid=candidate["params"],
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=1,
                    return_train_score=True
                )
            else:
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=candidate["params"],
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=1,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                    return_train_score=True
                )

            search.fit(X, y)

            print(f"Best score for {candidate['model'].__class__.__name__}: {search.best_score_}")

            self.scores_history.append(
                {"model": candidate['model'].__class__.__name__, "scores": search.cv_results_['mean_test_score']}
            )

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_estimator = search.best_estimator_

        if best_estimator is None:
            raise ValueError("No valid model was found during the search.")

        self.best_model = best_estimator

    def plot_convergence(self):
        """
        Wizualizacja zbieżności wyników dla różnych modeli.
        """
        plt.figure(figsize=(10, 6))
        for result in self.scores_history:
            plt.plot(
                range(1, len(result["scores"]) + 1),
                result["scores"],
                label=result["model"]
            )

        plt.xlabel("Iteration")
        plt.ylabel("Mean Test Score")
        plt.title("Convergence of Model Selection")
        plt.legend()
        plt.grid()
        plt.show()

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)

    def get_best_model(self):
        """
        Zwraca najlepszy wytrenowany model.
        """
        return self.best_model
