import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
import os

class ShapPLOT:
    """
    A class for generating SHAP (SHapley Additive exPlanations) visualizations
    and analyses for machine learning models. This includes support for summary
    plots, dependence plots, decision plots, and force plots, specifically
    tailored for binary classification and regression tasks.
    """
    @staticmethod
    def is_available(best_model, X_train):
        """
        Checks if SHAP is suitable for the given model and dataset.

        Parameters:
        - best_model: The trained machine learning model.
        - X_train: pandas DataFrame - The training dataset.

        Returns:
        - true (always but with potential warnings)
        """
        if best_model.__class__.__name__ in ["CatBoost","Neural Network", "Baseline"]:
            return False
        if X_train.shape[1] > 400:
            warnings.warn(
                "Too many columns for SHAP explanations - be aware of the performance impact"
            )
        
        if X_train.shape[0] < 30:
            warnings.warn(
                "Not enough records for SHAP explanations - be aware of the performance impact"
            )
        return True
    
    @staticmethod
    def get_explainer(best_model, X_train):
        """
        Creates a SHAP explainer instance based on the model type.

        Parameters:
        - best_model: The trained machine learning model.
        - X_train: pandas DataFrame - The training dataset.

        Returns:
        - shap.Explainer: The appropriate SHAP explainer for the model.
        """
        model_name = best_model.__class__.__name__
        shap_explainer = None
        if model_name in [
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "DecisionTreeRegressor",
            "RandomForestRegressor"
        ]:
            shap_explainer = shap.TreeExplainer(best_model)
        else:
            shap_explainer = shap.LinearExplainer(best_model, X_train)

        return shap_explainer
    
    @staticmethod
    def limit_df(X_test, y_test):
        """
        Limits the size of the test dataset to improve SHAP performance on large datasets.

        Parameters:
        - X_test: pandas DataFrame - The test dataset features.
        - y_test: pandas Series or numpy array - The test dataset target variable.

        Returns:
        - X_test_lim: pandas DataFrame - Limited test dataset features.
        - y_test_lim: pandas Series - Limited test dataset target variable.
        """
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test, name="target")
        ROW_LIMIT = 1000
        if X_test.shape[0] > ROW_LIMIT:
            X_test.reset_index(inplace=True, drop=True)
            y_test.reset_index(inplace=True, drop=True)
            X_test_lim = X_test.sample(ROW_LIMIT)
            y_test_lim = y_test[X_test_lim.index]
        else:
            X_test_lim = X_test
            y_test_lim = y_test
        return X_test_lim, y_test_lim
    
    
    def get_predictions(best_model, X_test_lim, y_test_lim):
        """
        Generates predictions and calculates residuals for regression tasks.

        Parameters:
        - best_model: The trained machine learning model.
        - X_test_lim: pandas DataFrame - Limited test dataset features.
        - y_test_lim: pandas Series - Limited test dataset target variable.

        Returns:
        - pandas DataFrame: A DataFrame containing residuals, prediction indices,
          and target values, sorted by residuals.
        """
        predictions = best_model.predict(X_test_lim)
        residuals = np.abs(np.array(y_test_lim) - predictions)
        pred_dataframe = pd.DataFrame(
                    {"res": residuals, "lp": range(residuals.shape[0]), "target": np.array(y_test_lim)},
                        index=X_test_lim.index,
                    )
        pred_dataframe = pred_dataframe.sort_values(by="res", ascending=False)
        return pred_dataframe
    
    @staticmethod
    def shap_summary_plot(shap_values, X_test_lim, best_model, results_dir, pdf=None):
        """
        Generates a SHAP summary plot to visualize feature importance.

        Parameters:
        - shap_values: numpy array - SHAP values for the test dataset.
        - X_test_lim: pandas DataFrame - Limited test dataset features.
        - best_model: The trained machine learning model.
        - model_file_path: string - Path to save the feature importance info based on shap summary plot as a CSV file.
        - pdf: matplotlib.backends.backend_pdf.PdfPages - PDF file to save plots (optional).
        """
        try:
            fig = plt.gcf()
            
            shap.summary_plot(
                shap_values, X_test_lim, show=False
            )
            plt.title(f"{best_model.__class__.__name__} SHAP summary plot")
            print(f"Global explanations - Shap Feature Importance of each feature in the dataset.")
            print("- Each dot represents one observation.")
            print("- The color of the dot indicates the feature value (red: high, blue: low).")
            print("- If the dot is on the right side of middle line, it means that it contributes to the higher prediction value.")
            print("- If the dot is on the left side of middle line, it means that it contributes to the lower prediction value.")
            print("- for example, if there is a blue cloud on the right side of the middle line, it means that low values of the feature\
                  \ncontribute to the higher prediction value.")
            plt.show()
            fig.tight_layout()  
            
            if pdf:
                pdf.savefig(fig)  
            plt.close(fig)

            vals = None
            if isinstance(shap_values, list):
                vals = None
                for sh in shap_values:
                    v = np.abs(sh).mean(0)
                    if vals is None:
                        vals = v
                    else:
                        vals += v
            else:    
                vals = np.abs(shap_values).mean(0)

            feature_importance = pd.DataFrame(
                list(zip(X_test_lim.columns, vals)), columns=["feature", "shap_importance"]
            )
            feature_importance.sort_values(
                by=["shap_importance"], ascending=False, inplace=True
            )

            feature_importance.to_csv(
                os.path.join(results_dir, f"{best_model.__class__.__name__}_shap_importance.csv"),
                index=False,
            )
        except Exception as e:
            print(f"Error in shap_summary_plot: {e}")
            
    @staticmethod
    def shap_dependence(shap_values, X_test_lim, pdf=None):
        """
        Creates SHAP dependence plots for the most important features.

        Parameters:
        - shap_values: numpy array - SHAP values for the test dataset.
        - X_test_lim: pandas DataFrame - Limited test dataset features.
        - pdf: matplotlib.backends.backend_pdf.PdfPages - PDF file to save plots (optional).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plt.figure(figsize=(14, 7))
            print(f"SHAP Dependence Plots - Relationship between feature values and their impact on predictions.")
            print("- X-axis: Feature values.")
            print("- Y-axis (left): SHAP values, showing the feature's contribution to the prediction.")
            print("- Y-axis (right): Other feature values (With the highest ranked importance pair).")
            print("- Points on the top indicate contribution to higher prediction values.")
            print("- Points on the bottom indicate contribution to lower prediction values.")
            print("- The color represents the value of another interacting feature.")
            print("- Patterns in the plots (e.g., slopes or clusters) show how the feature influences predictions.")
            plots_counter = np.min([9, X_test_lim.shape[1]])
            cols_cnt = 3
            rows_cnt = 3
            if plots_counter < 4:
                rows_cnt = 1
            elif plots_counter < 7:
                rows_cnt = 2
                 
            for i in range(plots_counter):
                ax = fig.add_subplot(rows_cnt, cols_cnt, i + 1)
                shap.dependence_plot(
                    f"rank({i})",
                    shap_values,
                    X_test_lim,
                    show=False,
                    title=f"Dependence plot - Importance #{i+1}",
                    ax=ax,
                )

            fig.tight_layout(pad=2.0)
            plt.show()
            if pdf:
                pdf.savefig(
                    fig
                )
            plt.close("all")
    
    @staticmethod    
    def explain_with_shap(best_model, X_train, X_test, y_test, ml_task, results_dir, pdf=None):
        """
        Main function to generate SHAP explanations for a given model and dataset.

        Parameters:
        - best_model: The trained machine learning model.
        - X_train: pandas DataFrame - The training dataset.
        - X_test: pandas DataFrame - The test dataset features.
        - y_test: pandas Series or numpy array - The test dataset target variable.
        - ml_task: string - Task type ('BINARY_CLASSIFICATION' or 'REGRESSION').
        - model_file_path: string - Path to save SHAP outputs.
        - pdf: matplotlib.backends.backend_pdf.PdfPages - PDF file to save plots (optional).
        """
        
        
        
        if not ShapPLOT.is_available(best_model, X_train):
            return
        
        explainer = ShapPLOT.get_explainer(best_model, X_train)
        
        X_test_lim, y_test_lim = ShapPLOT.limit_df(X_test, y_test)

        shap_values = explainer.shap_values(X_test_lim)

        expected_value = explainer.expected_value
        
        if ml_task == "BINARY_CLASSIFICATION" and isinstance(shap_values, np.ndarray):
            shap_values = shap_values[:,:,1]
            expected_value = expected_value[1]
        
        print(f"\nSHAP explanations:")
        
        ShapPLOT.shap_summary_plot(shap_values, X_test_lim, best_model, results_dir, pdf)
        
        ShapPLOT.shap_dependence(shap_values, X_test_lim, pdf) 
        
        df_preds = ShapPLOT.get_predictions(best_model, X_test_lim, y_test_lim)
        
        if ml_task == "BINARY_CLASSIFICATION":
            ShapPLOT.decisions_binary(df_preds, shap_values, expected_value, X_test_lim, y_test_lim, pdf) 
            ShapPLOT.forceplot_binary(best_model, shap_values, expected_value, X_test_lim, results_dir)
        else:
            ShapPLOT.decisions_regression(df_preds, shap_values, expected_value, X_test_lim, pdf)
            ShapPLOT.forceplot_regression(df_preds, shap_values, expected_value, X_test_lim, results_dir)



    @staticmethod
    def decisions_binary(
        df_preds,
        shap_values,
        expected_value,
        x_test_lim,
        y_test_lim,
        pdf,
    ):
        """
        Creates SHAP decision plots for binary classification.

        Parameters:
        - df_preds: pandas DataFrame - Predictions DataFrame with residuals and target.
        - shap_values: numpy array - SHAP values for the test dataset.
        - expected_value: float - SHAP baseline value.
        - x_test_lim: pandas DataFrame - Limited test dataset features.
        - y_test_lim: pandas Series - Limited test dataset target variable.
        - pdf: matplotlib.backends.backend_pdf.PdfPages - PDF file to save plots (optional).
        """
        
        print("- SHAP Decision Plots - These plots visualize the contribution of each feature to the model's prediction for individual observations.")
        print("- X-axis: Model output value.")
        print("- Y-axis: Path of the prediction, starting from the baseline (expected value) to the final prediction.")
        print("- The vertical position of each step represents the order in which the model uses the features.\n")
        
        for t in np.unique(y_test_lim):
            fig = plt.gcf()
            shap.decision_plot(
                expected_value,
                shap_values[df_preds[df_preds.target == t].lp[:10], :],
                x_test_lim.loc[df_preds[df_preds.target == t].index[:10]],
                show=False,
            )
            plt.title(f"SHAP decision plot for class {t} - worst decisions")
            print("This plot shows top 10 observations with highest residual values.")
            plt.show()
            fig.tight_layout(pad=2.0)
            if pdf:
                pdf.savefig(fig)
            plt.close("all")

            fig = plt.gcf()
            shap.decision_plot(
                expected_value,
                shap_values[df_preds[df_preds.target == t].lp[-10:], :],
                x_test_lim.loc[df_preds[df_preds.target == t].index[-10:]],
                show=False,
            )
            plt.title(f"SHAP decision plot for class {t} - best decisions")
            print("This plot shows top 10 observations with lowest residual values.")

            plt.show()
            fig.tight_layout(pad=2.0)
            if pdf:
                pdf.savefig(fig)
            plt.close("all")
    
    @staticmethod
    def decisions_regression(
        df_preds,
        shap_values,
        expected_value,
        x_test_lim,
        pdf,
    ):
        """
        Creates SHAP decision plots for regression tasks.

        Parameters:
        - df_preds: pandas DataFrame - Predictions DataFrame with residuals and target.
        - shap_values: numpy array - SHAP values for the test dataset.
        - expected_value: float - SHAP baseline value.
        - x_test_lim: pandas DataFrame - Limited test dataset features.
        - pdf: matplotlib.backends.backend_pdf.PdfPages - PDF file to save plots (optional).
        """
        print("- SHAP Decision Plots - These plots visualize the contribution of each feature to the model's prediction for individual observations.")
        print("- X-axis: Model output value.")
        print("- Y-axis: Path of the prediction, starting from the baseline (expected value) to the final prediction.")
        print("- The vertical position of each step represents the order in which the model uses the features.\n")
        
        fig = plt.gcf()
        shap.decision_plot(
            expected_value,
            shap_values[df_preds.lp[:10], :],
            x_test_lim.loc[df_preds.index[:10]],
            show=False,
        )
        plt.title("Decision plot - worst decisions")
        print("This plot shows top 10 observations with highest residual values.")
        plt.show()
        fig.tight_layout(pad=2.0)
        
        if pdf:
            pdf.savefig(fig)
        plt.close("all")

        fig = plt.gcf()
        shap.decision_plot(
            expected_value,
            shap_values[df_preds.lp[-10:], :],
            x_test_lim.loc[df_preds.index[-10:]],
            show=False,
        )
        plt.title("Decision plot - best decisions")
        print("This plot shows top 10 observations with lowest residual values.")
        plt.show()
        fig.tight_layout(pad=2.0)
        if pdf:
            pdf.savefig(fig)
        plt.close("all")
        
    @staticmethod
    def forceplot_binary(
        best_model,
        shap_values,
        expected_value,
        x_test_lim,
        results_dir,
    ):
        """
        Creates force plots for binary classification to explain predictions.

        Parameters:
        - best_model: The trained machine learning model.
        - shap_values: numpy array - SHAP values for the test dataset.
        - expected_value: float - SHAP baseline value.
        - x_test_lim: pandas DataFrame - Limited test dataset features.
        - model_file_path: string - Path to save the plots.
        """
        print("- SHAP Force Plot - This plot visualizes how each feature contributes to a specific prediction by a model.")
        print("- X-axis: The feature names and values, representing the impact of each feature on the model's output.")        
        print("- The color intensity indicates the magnitude and direction of each feature’s effect on the final prediction (red for positive, blue for negative).")
        print("- The baseline is represented by the initial model output, and the steps show how each feature either increases or decreases the prediction.")
        prob_class_1 = best_model.predict_proba(x_test_lim)[:, 1]
        prob_class_0 = best_model.predict_proba(x_test_lim)[:, 0]

        top_indices_class_1 = prob_class_1.argsort()[-1:]
        top_indices_class_0 = prob_class_0.argsort()[-1:]
        fig = plt.gcf()
        shap.plots.force(expected_value, shap_values[top_indices_class_1, :], x_test_lim.iloc[top_indices_class_1, :], matplotlib=True, show=False, plot_cmap="CyPU")
        plt.title("Force plot for class 1 - observation with biggest prediction probability")
        #plt.show()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                results_dir, f"force_plot_class_1.png"
            )
        )
        plt.show()
        plt.close("all")

        fig = plt.gcf()
        shap.plots.force(expected_value, shap_values[top_indices_class_0, :], x_test_lim.iloc[top_indices_class_0, :], matplotlib=True, show=False)
        plt.title("Force plot for class 0 - observation with biggest prediction probability")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                results_dir, f"force_plot_class_0.png"
            )
        )
        plt.show()
        plt.close("all")
    @staticmethod
    def forceplot_regression(
        df_preds,
        shap_values,
        expected_value,
        x_test_lim,
        results_dir,
    ):
        """
        Creates force plots for regression problem to explain predictions.

        Parameters:
        - shap_values: numpy array - SHAP values for the test dataset.
        - expected_value: float - SHAP baseline value.
        - x_test_lim: pandas DataFrame - Limited test dataset features.
        - model_file_path: string - Path to save the plots.
        """
        print("- SHAP Force Plots - These plots visualize how each feature contributes to a specific prediction by a model.")
        print("- X-axis: The feature names and values, representing the impact of each feature on the model's output.")        
        print("- The color intensity indicates the magnitude and direction of each feature’s effect on the final prediction (red for positive, blue for negative).")
        print("- The baseline is represented by the initial model output, and the steps show how each feature either increases or decreases the prediction.")
        
        fig = plt.gcf()
        shap.plots.force(expected_value, 
                        shap_values[df_preds.lp[-1:], :],
                        x_test_lim.loc[df_preds.index[-1:]], 
                         matplotlib=True,
                         show=False)        
        plt.title("Force plot for observation with the lowest residual value")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                results_dir, f"force_plot_regression_best.png"
            )
        )
        plt.show()
        plt.close("all")

        fig = plt.gcf()
        shap.plots.force(expected_value, 
                        shap_values[df_preds.lp[:1], :],
                        x_test_lim.loc[df_preds.index[:1]], 
                         matplotlib=True,
                         show=False)
        plt.title("Force plot for observation with the highest residual value")        
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                results_dir, f"force_plot_regression_worst.png"
            )
        )
        plt.show()
        plt.close("all")
        
        
