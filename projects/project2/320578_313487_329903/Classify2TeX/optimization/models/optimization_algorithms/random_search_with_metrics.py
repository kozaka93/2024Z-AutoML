from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, brier_score_loss
import pandas as pd
import numpy as np
import random

class RandomSearchWithMetrics:
    def __init__(self, pipeline, params, X, y, n_iter=10, cv=5, random_state=42, n_repeats=5):
        """
        Initialize the RandomSearchWithMetrics class.

        Args:
            pipeline: The ML pipeline (e.g., sklearn Pipeline object).
            params: Dictionary of hyperparameter names and their possible values.
            X: Feature dataset (numpy array or Pandas DataFrame).
            y: Target dataset (numpy array or Pandas Series).
            n_iter: Number of iterations to perform random search (default 10).
            cv: Number of cross-validation splits (default 5).
            random_state: Random seed for reproducibility (default 42).
            n_repeats: Number of times to repeat cross-validation for stability (default 5).
        """
        self.pipeline = pipeline
        self.params = params
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.n_repeats = n_repeats  # For repeated cross-validation to ensure stable metrics
        self.history = pd.DataFrame()  # In-memory storage for the results
        self.classifiers = []

    def generate_random_params(self):
        """
        Randomly generate a set of hyperparameters from the provided parameter grid.

        Returns:
            A dictionary with randomly selected values for each parameter.
        """
        params = {}
        for key, values in self.params.items():
            if isinstance(values, list):  # Ensure values is a list to allow random selection
                params[key] = random.choice(values)  # Select a random value from the list of possible values
        return params

    def fit_and_evaluate(self):
        """
        Perform random search with cross-validation and store the results in `self.history`.
        This method will perform the following:
            - Randomly select hyperparameters for each iteration.
            - Apply them to the pipeline.
            - Perform cross-validation `n_repeats` times to calculate stability in metrics.
            - Compute F1 score, accuracy, and ROC AUC.
        """
        # Set seed for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        for i in range(self.n_iter):  # Perform `n_iter` random search iterations
            # Randomly generate a new set of hyperparameters
            params = self.generate_random_params()
            self.pipeline.set_params(**params)  # Apply the hyperparameters to the pipeline

            # Initialize lists to accumulate metrics across repeats
            f1_scores, accuracies, roc_aucs = [], [], []

            for j in range(self.n_repeats):  # Repeat cross-validation `n_repeats` times for stability
                # Create KFold object for cross-validation (shuffle=True for random splits)
                kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

                # Perform cross-validation predictions for both labels and probabilities
                y_pred = cross_val_predict(self.pipeline, self.X, self.y, cv=kf, method='predict')
                y_probabilities = cross_val_predict(self.pipeline, self.X, self.y, cv=kf, method='predict_proba')
                y_probabilities = y_probabilities[:, 1]  # Get probabilities for the positive class (binary classification)

                # Calculate and append evaluation metrics for binary classification
                f1_scores.append(f1_score(self.y, y_pred, average='weighted'))  # Weighted F1 score
                accuracies.append(accuracy_score(self.y, y_pred))  # Accuracy score
                roc_aucs.append(roc_auc_score(self.y, y_probabilities))  # ROC AUC score

                # Train the model on the entire dataset
                self.pipeline.fit(self.X, self.y)
                self.classifiers.append(self.pipeline.named_steps['clf'])  # Append the classifier to the list

            # Calculate average metrics across all repeats
            avg_metrics = {
                'f1': np.mean(f1_scores),  # Average F1 score
                'accuracy': np.mean(accuracies),  # Average accuracy
                'roc_auc': np.mean(roc_aucs)  # Average ROC AUC
            }

            # Print the current results for the user (useful for monitoring)
            print("Checked another model, results using cross-validation:", avg_metrics)

            # Add the hyperparameter values to the metrics dictionary
            avg_metrics.update(params)

            # Append the results to the history DataFrame
            self.history = pd.concat([self.history, pd.DataFrame([avg_metrics])], ignore_index=True)

    def get_results(self):
        """
        Retrieve all combinations of hyperparameters and their corresponding metric values
        (F1 score, accuracy, and ROC AUC).

        Returns:
            pd.DataFrame: DataFrame with hyperparameter combinations and metrics.
        """
        return self.history, self.classifiers  # Return the history DataFrame containing the results, and the classifiers

