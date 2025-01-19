from evaulator import Evaluator
from optimizers import Optimizer
from preprocess import DataPreprocessor

class AutoMLClassifier:
    def __init__(
        self,
        score_metric='accuracy',
        optimizer='bayes',
        n_iterations=20,
        cv=5,
        n_jobs=-1,
        seed=None,
        missing_strategy='mean',
        normalization_method='standard',
        num_features=None,
    ):
        self.optimizer = Optimizer(score_metric, optimizer, cv, n_iterations, seed, n_jobs)
        self.evaluator = Evaluator(score_metric)
        self.preprocessor = DataPreprocessor(missing_strategy, normalization_method, num_features)

        self.best_model = None
        self.best_models = None
        self.best_params = None
        self.best_score = None
        self.scores = None

        self.preprocessed_data_ = None
        self.target_ = None
        self.feature_names_ = None  # Save feature names after preprocessing

    def fit(self, X, y):
        # Perform preprocessing and save feature names
        self.preprocessed_data_ = self.preprocessor.preprocess_pipeline(X, target=y)
        # Ensure feature names include transformed columns (after one-hot encoding)
        self.feature_names_ = self.preprocessed_data_.columns.tolist()
        self.target_ = y

        # Optimize model and save results
        self.best_model, self.best_params, self.best_score = self.optimizer.optimize(self.preprocessed_data_, y)
        self.scores = self.optimizer.scores
        self.best_models = self.optimizer.best_models


    def predict(self, X):
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        # Apply preprocessing to align with training
        X = self.preprocessor.preprocess_pipeline(X)
        # Align test data columns with training columns
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        X = self.preprocessor.preprocess_pipeline(X)
        # Align test data columns with training columns
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X)
        else:
            raise AttributeError(f"The selected model ({type(self.best_model).__name__}) does not support probability predictions.")


    def evaluate(self, metric=None):
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        return self.evaluator.evaluate(self.best_model, self.preprocessed_data_, self.target_, metric)

    def generate_report(self):
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        return self.evaluator.generate_report(self.best_model, self.scores, self.preprocessed_data_, self.target_)
