from .preprocessing import DataPreprocessor
from .model_selection import ModelSelector
from .evaluation import EvaluationReports


class AutoMLClassifier:
    """
    Klasa integrująca preprocessing, model selection i ewaluację.
    """

    def __init__(self, **kwargs):
        self.preprocessor = DataPreprocessor(
            strategy_num=kwargs.get("strategy_num", "mean"),
            strategy_cat=kwargs.get("strategy_cat", "most_frequent")
        )
        self.model_selector = ModelSelector(
            search_method=kwargs.get("search_method", "grid"),
            scoring=kwargs.get("scoring", "accuracy"),
            n_iter=kwargs.get("n_iter", 10),
            cv=kwargs.get("cv", 3),
            random_state=kwargs.get("random_state", 42)
        )
        self.evaluator = EvaluationReports()

    def fit(self, X, y):
        """
        Trening pipeline'u.
        """
        self.preprocessor.fit(X)
        X_preprocessed = self.preprocessor.transform(X)

        self.model_selector.fit(X_preprocessed, y)

        if self.model_selector.best_model is None:
            raise ValueError("Training failed. No valid model was selected.")

    def evaluate(self, X_test, y_test, report_path="report.html"):
        """
        Ewaluacja i generowanie raportu.
        """
        X_test_preprocessed = self.preprocessor.transform(X_test)
        report = self.evaluator.evaluate(self.model_selector.best_model, X_test_preprocessed, y_test)
        self.evaluator.save_html_report(report, report_path)

    def plot_convergence(self):
        """
        Wizualizacja zbieżności wyników modeli w trakcie treningu.
        """
        self.model_selector.plot_convergence()
