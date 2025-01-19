from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from .optimization_algorithms.random_search_with_metrics import RandomSearchWithMetrics


class XGBoostRandomSearch:
    def __init__(self, dataset, n_iter=10, cv=5, random_state=42, n_repeats=5):
        """
        Initialize the XGBoostRandomSearch class.

        Args:
            dataset: The preprocessed dataset (Pandas DataFrame). Assumes 'target' as the label column.
            n_iter: Number of iterations to perform random search (int).
            cv: Number of cross-validation folds (int).
            random_state: Random seed for reproducibility (int).
            n_repeats: Number of times to repeat cross-validation for stability (int).
        """
        # Extract features (X) and target (y) from the dataset
        self.y = dataset['target']  # Target variable
        self.X = dataset.drop(columns=['target']) 
        self.history = None  # Store results history
        self.random_state = random_state  # Random seed for reproducibility

        # Define the pipeline: contains only the XGBoost classifier
        self.pipeline = Pipeline([
            ('clf', XGBClassifier(random_state=self.random_state, objective='binary:logistic'))  # Binary classification
        ])

        # Define the hyperparameter search space for XGBoost
        self.params = {
            'clf__eval_metric': ['logloss'],  # Evaluation metric: log-loss for binary classification
            'clf__n_estimators': [50, 100, 200, 500],  # Number of boosting rounds
            'clf__max_depth': [3, 6, 10, 15],  # Maximum depth of trees
            'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
            'clf__subsample': [0.5, 0.7, 0.9, 1.0],  # Fraction of samples used per tree
            'clf__colsample_bytree': [0.5, 0.7, 0.9, 1.0],  # Fraction of features used per tree
            'clf__min_child_weight': [1, 3, 5, 7],  # Minimum child weight for leaf nodes
            'clf__gamma': [0, 0.1, 0.2, 0.3],  # Minimum loss reduction to split nodes
            'clf__reg_alpha': [0, 0.01, 0.1, 1],  # L1 regularization strength
            'clf__reg_lambda': [1, 1.5, 2, 5]  # L2 regularization strength
        }

        # Initialize custom random search with metrics evaluation
        self.random_search = RandomSearchWithMetrics(
            pipeline=self.pipeline,  # Pipeline with XGBoost
            params=self.params,  # Search space
            X=self.X,  # Features
            y=self.y,  # Target
            n_iter=n_iter,  # Number of random search iterations
            cv=cv,  # Cross-validation folds
            random_state=random_state,  # Random seed for reproducibility
            n_repeats=n_repeats  # Repeated CV for stability
        )

        self.classifiers = []

    def perform_random_search(self):
        """
        Perform hyperparameter tuning using random search.

        Returns:
            A Pandas DataFrame containing the results of the random search.
        """
        # Fit and evaluate the model with different hyperparameter combinations
        self.random_search.fit_and_evaluate()

        # Retrieve and store the results from random search
        self.history, random_search_classifiers = self.random_search.get_results()
        for clf in random_search_classifiers:
            self.classifiers.append(clf)
        return self.history

    def fit_and_evaluate_default(self):
        """
        Fit and evaluate the XGBoost model using default hyperparameters.

        Returns:
            A Pandas DataFrame containing metrics and hyperparameters of the default model.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        
        # Initialize the XGBoost classifier with default parameters
        clf = XGBClassifier(random_state=42, objective='binary:logistic')

        # Fit the classifier on the training data
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)  # Predicted labels
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None  # Predicted probabilities

        # Calculate evaluation metrics
        f1 = f1_score(y_test, y_pred, average="weighted")  # F1 score
        accuracy = accuracy_score(y_test, y_pred)  # Accuracy
        roc_auc = roc_auc_score(y_test, y_pred_proba, average="weighted") if y_pred_proba is not None else None  # ROC AUC

        # Prepare results with hyperparameters and metrics
        results = {
            'f1': f1,  # Weighted F1 score
            'accuracy': accuracy,  # Accuracy score
            'roc_auc': roc_auc,  # ROC AUC score
            'clf__eval_metric': 'logloss',  # Default eval metric for classification
            'clf__n_estimators': 100,  # Default: 100 boosting rounds
            'clf__max_depth': 6,  # Default: Maximum tree depth
            'clf__learning_rate': 0.3,  # Default: Learning rate
            'clf__subsample': 1.0,  # Default: Use all samples per tree
            'clf__colsample_bytree': 1.0,  # Default: Use all features per tree
            'clf__min_child_weight': 1,  # Default: Minimum child weight
            'clf__gamma': 0,  # Default: No split pruning
            'clf__reg_alpha': 0,  # Default: No L1 regularization
            'clf__reg_lambda': 1,  # Default: Default L2 regularization
        }

        # Display the results for the default model
        print("Default model results:", results)
        self.classifiers.append(clf)  # Append the default classifier to the list of classifiers

        # Convert results to a Pandas DataFrame
        self.default_results = pd.DataFrame([results])

        return self.default_results

    def get_results(self):
        """
        Retrieve results from both the default model and random search.

        Returns:
            A Pandas DataFrame with results from both default and tuned models.
        """
        # Fit and evaluate the default model
        default_results = self.fit_and_evaluate_default()

        # Perform random search and get tuned results
        random_results = self.perform_random_search()

        # Combine default and random search results into a single DataFrame
        res = pd.concat([default_results, random_results], ignore_index=True)

        return res, self.classifiers
