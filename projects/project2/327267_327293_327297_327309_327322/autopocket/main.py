import pandas as pd

from autopocket.preprocessing.Preprocessor import  Preprocessor
from autopocket.algorithms.Modeller import Modeller
from autopocket.postprocessing.Postprocessor import Postprocessor


class AutoPocketor():
    def __init__(self):
        """
        Class for performing AutoML tasks.
        """
        pass

    def doJob(self,
              path,
              target,
              num_strategy='mean', 
              cat_strategy='most_frequent', 
              fill_value=None,
              generate_shap=True,
              generate_lime_pdp_ice=True,
              features_for_displaying_plots=None,
              subset_fraction_for_ICE_plot=None,
              additional_estimators=None
              ):
        """
        Main method for performing AutoML tasks including preprocessing, modeling, and postprocessing.

        Parameters:
            path: str
                Path to the dataset file.
            target: str
                Name of the target column in the dataset.
            num_strategy: string, optional, default='mean'
                The imputation strategy for numerical columns ('mean', 'median', 'most_frequent', 'constant').
            cat_strategy: string, optional, default='most_frequent'
                The imputation strategy for categorical columns ('most_frequent', 'constant').
            fill_value: Any, optional, default=None
                The value to use when filling missing values, if 'constant' strategy is used.
            generate_shap: bool, optional, default=True
                Whether to generate SHAP explanations in postprocessing.
            generate_lime_pdp_ice: bool, optional, default=True
                Whether to generate LIME explanations, Partial Dependence Plots (PDPs) and 
                Individual Conditional Expectation (ICE) plots in postprocessing.
            features_for_displaying_plots: list, optional, default=None
                List of non-binary features for which PDP and ICE plots will be displayed. If None, top non-binary 
                features will be selected automatically based on LIME feature importance.
            subset_fraction_for_ICE_plot: float, optional, default=None
                Fraction of rows to use for generating ICE plots. If None, all rows are used.

        Workflow:
            1. **Preprocessing**:
                - Prepares the data by handling missing values, inconsistent formats, and feature encoding.
                - Outputs cleaned data (`X`) and target (`y`), and determines the type of machine learning task (e.g., classification or regression).
            
            2. **Modeling**:
                - Trains multiple models using scikit-learn.
                - Selects the best-performing model based on a predefined metric.
                - Returns the best model, evaluation metric, and all trained estimators for further analysis.

            3. **Postprocessing**:
                - Provides model interpretability and visualization.
                - Generates SHAP and LIME explanations, PDPs, and ICE plots (if enabled).
                - Saves results and visualizations in structured outputs (e.g., PDFs, JSON).

        Outputs:
            - Printed logs of each step, including preprocessing, modeling, and postprocessing progress.
            - Visualizations and explanations saved to the `results` directory.
        """

        print("Performing preprocessing...")
        X, y, ml_type = Preprocessor().preprocess(path = path, 
                                                  target=target, 
                                                  num_strategy=num_strategy, 
                                                  cat_strategy=cat_strategy, 
                                                  fill_value=fill_value)
        print("X shape:", X.shape)
        print("Preprocessing done.")

        print("\nPerforming modelling...")
        best_model, results_dir = Modeller(additional_estimators=additional_estimators).model(X, 
                                                                                              y, 
                                                                                              ml_type)
        print(f"Chosen model: {best_model.__class__.__name__}")
        print("Modelling done.")

        X = pd.DataFrame(X)
        
        print("\nPerforming postprocessing...")
        Postprocessor(results_dir=results_dir).postprocess(best_model, 
                                                           X, 
                                                           y, 
                                                           ml_type, 
                                                           generate_shap, 
                                                           generate_lime_pdp_ice, 
                                                           features_for_displaying_plots, 
                                                           subset_fraction_for_ICE_plot)
        print("Postprocessing done.")
