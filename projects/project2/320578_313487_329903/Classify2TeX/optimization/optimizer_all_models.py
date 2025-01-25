from .models.random_forest_random_search import RandomForestRandomSearch
from .models.xgboost_random_search import XGBoostRandomSearch
from .models.decision_tree_random_search import DecisionTreeRandomSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd

class OptimizerAllModels:
    def __init__(self, dataset, random_state=42, n_iter=[0, 0, 0], cv=5, n_repeats=1, metric_to_eval = 'roc_auc'):
        """
        Initialize the Fit_all_models class.

        Args:
            dataset: The preprocessed dataset (Pandas DataFrame).
            random_state: Random seed for reproducibility.
            n_iter: Number of iterations to perform random search.
            cv: Number of cross-validation splits.
            random_state: Random seed for reproducibility.
            n_repeats: Number of times to repeat cross-validation for stability.
            metric_to_eval: Metric according to which the evaluation will be performed, possible values (roc_auc, f1, accuracy)
        """
        self.dataset = dataset
        self.test_size = 0.2 # Default test size
        self.random_state = random_state
        self.n_iter = n_iter
        self.cv = cv
        self.n_repeats = n_repeats
        self.metric_to_eval = metric_to_eval # Metric according to which the evaluation will be performed

        # Split the dataset into features (X) and target (y)
        self.y = dataset['target']  # Assumes 'target' column is the label
        self.X = dataset.drop(columns=['target']) # Drop the target column in the training set

        self.X_train = self.X.copy()
        self.y_train = self.y.copy()

        # Placeholder for hyperparameters and their metrics for all models

        self.params_rf = None

        self.params_dt = None

        self.params_xgb = None

        self.best_xgb_instance = None
        self.best_rf_instance = None
        self.best_dt_instance = None

    def tune_hyperparameters(self):
        """
        Perform hyperparameter tuning using DecisionTreeClassifierRandomSearch/RandomForestClassifier/XGBoostClassifier.

        Args:
            n_iter: Number of iterations to perform random search.
            cv: Number of cross-validation splits.
            random_state: Random seed for reproducibility.
            n_repeats: Number of times to repeat cross-validation for stability.
        """

        # Use the DecisionTreeClassifierRandomSearch class
        tuner_decision_tree = DecisionTreeRandomSearch(
            dataset=pd.concat([self.X_train, self.y_train], axis=1),
            n_iter=self.n_iter[0],
            cv=self.cv,
            random_state=self.random_state,
            n_repeats=self.n_repeats
        )

        # Use the RandomForestRandomSearch class
        tuner_rand_forest = RandomForestRandomSearch(
            dataset=pd.concat([self.X_train, self.y_train], axis=1),
            n_iter=self.n_iter[1],
            cv = self.cv,
            random_state=self.random_state,
            n_repeats=self.n_repeats

        )

        # Use the XGBoostRandomSearch class
        tuner_xgboost = XGBoostRandomSearch(
            dataset=pd.concat([self.X_train, self.y_train], axis=1),
            n_iter=self.n_iter[2],
            cv = self.cv,
            random_state=self.random_state,
            n_repeats=self.n_repeats
        )

        # Perform hyperparameter optimization using RandomSearch

        print("---Performing hyperparameter tuning for DecisionTreeClassifier...")
        self.params_dt, self.all_clf_dt = tuner_decision_tree.get_results()

        print("---Performing hyperparameter tuning for RandomForestClassifier...")
        self.params_rf, self.all_clf_rf = tuner_rand_forest.get_results()

        print("---Performing hyperparameter tuning for XGBoostClassifier...")
        self.params_xgb, self.all_clf_xgb = tuner_xgboost.get_results()

    def get_best_results(self):
        """
        Get the best hyperparameter combination and metrics for each model as a single-row DataFrame.
        Also, print the best hyperparameters and metrics for each model.
        """
        # Extract best results for each model
        best_rf = self.params_rf.loc[self.params_rf[self.metric_to_eval].idxmax()]
        # save the best model instance
        self.best_rf_instance = self.all_clf_rf[self.params_rf[self.metric_to_eval].idxmax()]

        best_dt = self.params_dt.loc[self.params_dt[self.metric_to_eval].idxmax()]
        # save the best model instance
        self.best_dt_instance = self.all_clf_dt[self.params_dt[self.metric_to_eval].idxmax()]

        best_xgb = self.params_xgb.loc[self.params_xgb[self.metric_to_eval].idxmax()]
        # save the best model instance
        self.best_xgb_instance = self.all_clf_xgb[self.params_xgb[self.metric_to_eval].idxmax()]

        # Print the best results
        print("The best hyperparameters for RandomForestClassifier are:")
        print(best_rf.to_frame().T, "\n")  # Print as a DataFrame row

        print("The best hyperparameters for DecisionTreeClassifier are:")
        print(best_dt.to_frame().T, "\n")  # Print as a DataFrame row

        print("The best hyperparameters for XGBoostClassifier are:")
        print(best_xgb.to_frame().T, "\n")  # Print as a DataFrame row

        # Combine all results into a single DataFrame
        best_results_df = pd.DataFrame({
            "Model": ["RandomForest", "DecisionTree", "XGBoost"],
            "Best_Hyperparameters_and_Metrics": [best_rf.to_dict(), best_dt.to_dict(), best_xgb.to_dict()]
        })
        
        return best_results_df
    
    def perform_analysis(self):

        self.tune_hyperparameters()

        return self.get_best_results()