from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from .optimization_algorithms.random_search_with_metrics import RandomSearchWithMetrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

class RandomForestRandomSearch:
    def __init__(self, dataset, n_iter=10, cv=5, random_state=42, n_repeats=5):
        """
        Initialize the RandomForestRandomSearch class.

        Args:
            dataset (pd.DataFrame): The preprocessed dataset containing features and the target column.
            n_iter (int): Number of iterations for random search.
            cv (int): Number of cross-validation splits.
            random_state (int): Random seed for reproducibility.
            n_repeats (int): Number of times to repeat cross-validation for stability.
        """
        # Separate features (X) and target variable (y)
        self.X = dataset.drop(columns=['target'])  # Assumes 'target' is the column name for labels
        self.y = dataset['target']  # Target variable
        self.history = None  # Stores results of random search
        self.random_state = random_state  # Seed for reproducibility

        # Define a pipeline with a RandomForestClassifier (additional preprocessing steps can be added here)
        self.pipeline = Pipeline([
            ('clf', RandomForestClassifier(random_state=random_state))  # Random forest classifier
        ])

        # Define hyperparameter search space for RandomForestClassifier
        self.params = {
            'clf__n_estimators': [50, 100, 200, 500],  # Number of trees in the forest
            'clf__criterion': ['gini', 'entropy', 'log_loss'],  # Split quality metrics
            'clf__max_depth': [None, 10, 20, 30, 40],  # Maximum depth of the tree
            'clf__min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
            'clf__min_samples_leaf': [1, 2, 4],  # Minimum samples required to be a leaf node
            'clf__min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1],  # Minimum fraction of weights for a leaf
            'clf__max_features': [None, 'sqrt', 'log2'],  # Number of features to consider for splits
            'clf__bootstrap': [True, False]  # Whether to use bootstrap samples
        }

        # Initialize the custom RandomSearchWithMetrics for hyperparameter tuning
        self.random_search = RandomSearchWithMetrics(
            pipeline=self.pipeline,  # Machine learning pipeline
            params=self.params,  # Hyperparameter search space
            X=self.X,  # Features
            y=self.y,  # Target variable
            n_iter=n_iter,  # Number of search iterations
            cv=cv,  # Number of cross-validation splits
            random_state=random_state,  # Seed for reproducibility
            n_repeats=n_repeats  # Stability through repeated cross-validation
        )

        self.classifiers = []

    def perform_random_search(self):
        """
        Perform random search for hyperparameter tuning using RandomSearchWithMetrics.

        Returns:
            pd.DataFrame: History of hyperparameter search results.
        """
        # Execute the random search and store results
        self.random_search.fit_and_evaluate()
        self.history, random_search_classifiers = self.random_search.get_results()
        for clf in random_search_classifiers:
            self.classifiers.append(clf)
        return self.history

    def fit_and_evaluate_default(self):
        """
        Fit and evaluate the RandomForestClassifier using default parameters.

        Returns:
            pd.DataFrame: Evaluation metrics and model parameters for the default configuration.
        """
        # Split dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )

        # Initialize RandomForestClassifier with default parameters
        clf = RandomForestClassifier(random_state=self.random_state)

        # Train the model on the training set
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        # Calculate evaluation metrics
        f1 = f1_score(y_test, y_pred, average="weighted")  # Weighted F1 score
        accuracy = accuracy_score(y_test, y_pred)  # Accuracy score
        roc_auc = (roc_auc_score(y_test, y_pred_proba, average="weighted")
                   if y_pred_proba is not None else None)  # ROC-AUC score (if available)

        # Compile results into a dictionary
        results = {
            'f1': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'clf__n_estimators': clf.n_estimators,
            'clf__criterion': clf.criterion,
            'clf__max_depth': clf.max_depth,
            'clf__min_samples_split': clf.min_samples_split,
            'clf__min_samples_leaf': clf.min_samples_leaf,
            'clf__min_weight_fraction_leaf': clf.min_weight_fraction_leaf,
            'clf__max_features': clf.max_features,
            'clf__bootstrap': clf.bootstrap
        }

        print("Default model results:", results)

        # Convert results to a DataFrame for easier comparison
        self.default_results = pd.DataFrame([results])
        self.classifiers.append(clf)
        return self.default_results

    def get_results(self):
        """
        Aggregate results from the default configuration and random search.

        Returns:
            pd.DataFrame: Combined results of default and tuned configurations.
        """
        # Evaluate default configuration
        default_results = self.fit_and_evaluate_default()

        # Perform random search for hyperparameter tuning
        random_results = self.perform_random_search()

        # Combine both results into a single DataFrame
        res = pd.concat([default_results, random_results], ignore_index=True)
        return res, self.classifiers
