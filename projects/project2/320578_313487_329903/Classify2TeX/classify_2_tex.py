from .optimization.optimizer_all_models import OptimizerAllModels
from .preprocessing.data_preprocessor import DataPreprocessor
from .report.report_generator import ReportGenerator
from .xai.explain_decision_tree import ExplainDecisionTree

class Classify2TeX:
    def __init__(self, dataframe, target_column_name, test_size=0.2, random_state=42, n_iter=[0, 0, 0], cv=5, n_repeats=1, metric = 'roc_auc'):
        """
        Initialize the Auto2Class for automated binary classification model selection.

        Args:
            dataframe: The dataset to be used for training and evaluation.
            target_column_name: The name of the column containing the target variable.
            test_size: Fraction of the data to be used for testing (default is 0.2).
            random_state: Random seed for reproducibility.
            n_iter: Number of iterations for model optimization.
            cv: Number of cross-validation splits.
            n_repeats: Number of times to repeat cross-validation for stability.
            metric: The evaluation metric be optimized during model selection (default is 'roc_auc').
        """
        self.dataframe = dataframe
        self.target_column_name = target_column_name
        self.n_iter = n_iter
        self.n_repeats = n_repeats
        self.metric = metric
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.params_rf = None
        self.params_dt = None
        self.params_xgb = None

        if len(self.n_iter) != 3:
            raise ValueError("n_iter should be a list of length 3.")
        
        for i in self.n_iter:
            if i < 0:
                raise ValueError("n_iter should be greater than or equal to 0.")
        

    def perform_model_selection(self):
        """
        Perform the model selection process by preprocessing the data and running optimization 
        on Random Forest, Decision Tree, and XGBoost models.

        This method preprocesses the data, performs model optimization, and stores the best 
        hyperparameters for each model.
        """
        # Preprocess the data
        preprocessed_data = DataPreprocessor(self.dataframe, self.target_column_name).preprocess()
        self.optimizer = OptimizerAllModels(preprocessed_data, self.random_state, self.n_iter, self.cv, self.n_repeats, self.metric)
        self.optimizer.perform_analysis()

        # Store the best hyperparameters for each model after optimization
        self.params_rf = self.optimizer.params_rf
        self.params_dt = self.optimizer.params_dt
        self.params_xgb = self.optimizer.params_xgb
        return 
    
    def generate_report(self, dataset_name):
        """
        Generate a report containing the results and information about dataeset.
        """
        if self.optimizer is None:
            raise ValueError("Model selection has not been performed. Please run perform_model_selection() first.")

        self.report_generator = ReportGenerator(self.dataframe, dataset_name, self.optimizer)
        self.report_generator.generate_report()

        print("Report generated successfully.")
        return
    
    def build_best_decision_tree(self):
        print("Below you can see how the best Decision Tree model takes decisions on each node.")
        explainer_dt = ExplainDecisionTree(self.optimizer.best_dt_instance)
        tree = explainer_dt.build_tree(self.optimizer.X_train, self.optimizer.y_train)
        return tree