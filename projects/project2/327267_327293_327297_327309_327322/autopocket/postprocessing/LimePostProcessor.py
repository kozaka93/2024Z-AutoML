import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings 
from autopocket.postprocessing.utils import is_uncorrelated
from autopocket.postprocessing.utils import normalize_feature_name

warnings.filterwarnings("ignore", category=FutureWarning, module="lime")


class LimePostprocessor():
    
    def __init__(self):
        """
        Initialize the LimePostprocessor class.
        """
        pass

    def explain_top_observations_with_lime(self, model, X_train, X_test, ml_type, num_features=10, pdf=None):
        """
        Generate LIME explanations for the two most influential observations for both classes (binary classification)
        or the two highest and lowest predictions (regression). Save all plots to the provided PDF file.

        Parameters:
            model: The trained model to explain.
            X_train: pd.DataFrame, training input features.
            X_test: pd.DataFrame, testing input features.
            ml_type: str, either "BINARY_CLASSIFICATION" or "LINEAR_REGRESSION".
            num_features: int, number of features to display in the explanation.
            pdf: PdfPages object to save plots. If None, no plots are saved.
        """
        explainer = LimeTabularExplainer(
            training_data=X_train.to_numpy(),
            feature_names=X_train.columns.tolist(),
            class_names=["Negative", "Positive"] if ml_type == "BINARY_CLASSIFICATION" else ["Target"],
            mode="classification" if ml_type == "BINARY_CLASSIFICATION" else "regression"
        )

        explanations = []

        print("\n LIME explanations:")
        print("- Each bar in the LIME plot represents the contribution of a specific feature to the prediction.")
        print("- Feature labels, such as 'feature_name < threshold' or 'feature_name > threshold', describe the range of values.")
        print("- These ranges are created by LIME's binning process, which discretizes continuous features into intervals.")
        print("- For example, 'age < 30' means that for this observation, the age being less than 30 contributes as shown.")
        print("- Bars pointing to the right (positive contributions) indicate that the feature pushes the prediction towards a positive outcome or higher value.")
        print("- Bars pointing to the left (negative contributions) indicate that the feature pulls the prediction towards a negative outcome or lower value.")
        print("- The length of the bar reflects the magnitude of the feature's influence on the model's prediction.")


        if ml_type == "BINARY_CLASSIFICATION":
            prob_class_1 = model.predict_proba(X_test)[:, 1]
            prob_class_0 = model.predict_proba(X_test)[:, 0]

            top_indices_class_1 = prob_class_1.argsort()[-2:]
            top_observations_class_1 = X_test.iloc[top_indices_class_1]

            print("Local explanantions - LIME explanations for the top 2 observations most likely to be class 1 (most influential):")
            for i, index in enumerate(top_indices_class_1):
                exp = explainer.explain_instance(
                    data_row=top_observations_class_1.iloc[i],
                    predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns)),
                    num_features=num_features
                )
                explanations.append(exp)
                fig = exp.as_pyplot_figure()
                plt.title(f"LIME Explanation for Top Class 1 Instance {i + 1}")

                plt.show()
                fig.set_size_inches(10, 6)
                fig.tight_layout()

                if pdf:
                    pdf.savefig(fig)
                plt.close(fig)

            top_indices_class_0 = prob_class_0.argsort()[-2:]
            top_observations_class_0 = X_test.iloc[top_indices_class_0]

            print("Local explanations - LIME explanations for the top 2 observations most likely to be class 0 (most influential):")
            for i, index in enumerate(top_indices_class_0):
                exp = explainer.explain_instance(
                    data_row=top_observations_class_0.iloc[i],
                    predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns)),
                    num_features=num_features
                )
                explanations.append(exp)
                fig = exp.as_pyplot_figure()
                plt.title(f"LIME Explanation for Top Class 0 Instance {i + 1}")

                plt.show()
                fig.set_size_inches(10, 6)
                fig.tight_layout()

                if pdf:
                    pdf.savefig(fig)
                plt.close(fig)

        elif ml_type == "LINEAR_REGRESSION":
            predictions = model.predict(X_test)

            top_indices_high = predictions.argsort()[-2:]
            top_observations_high = X_test.iloc[top_indices_high]

            print("Local explanations - LIME explanations for the top 2 highest predictions (in terms of value of the predicted feature):")
            for i, index in enumerate(top_indices_high):
                exp = explainer.explain_instance(
                    data_row=top_observations_high.iloc[i],
                    predict_fn=lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns)),
                    num_features=num_features
                )
                explanations.append(exp)
                fig = exp.as_pyplot_figure()
                plt.title(f"LIME Explanation for Top High Prediction Instance {i + 1}")

                plt.show()
                fig.set_size_inches(10, 6)
                fig.tight_layout()

                if pdf:
                    pdf.savefig(fig)
                plt.close(fig)

            top_indices_low = predictions.argsort()[:2]
            top_observations_low = X_test.iloc[top_indices_low]

            print("Local explanations - LIME explanations for the top 2 lowest predictions (in terms of value of the predicted feature):")
            for i, index in enumerate(top_indices_low):
                exp = explainer.explain_instance(
                    data_row=top_observations_low.iloc[i],
                    predict_fn=lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns)),
                    num_features=num_features
                )
                explanations.append(exp)
                fig = exp.as_pyplot_figure()
                plt.title(f"LIME Explanation for Top Low Prediction Instance {i + 1}")

                plt.show()
                fig.set_size_inches(10, 6)
                fig.tight_layout()

                if pdf:
                    pdf.savefig(fig)
                plt.close(fig)

        else:
            raise ValueError("Unsupported ml_type. Please use 'BINARY_CLASSIFICATION' or 'LINEAR_REGRESSION'.")

        return explanations



    def lime_summary_plot(self, explanations, max_features=15, pdf=None):
        """
        Generate a summary plot from multiple LIME explanations.

        Parameters:
            explanations: list, LIME explanation objects.
            max_features: int, maximum number of features to display in the plot.
            pdf: PdfPages object to save plots. If None, no plots are saved.
        """
        feature_importances = {}

        for exp in explanations:
            for feature, weight in exp.as_list():
                feature_importances[feature] = feature_importances.get(feature, 0) + abs(weight)

        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        sorted_features = sorted_features[:max_features]
        features, importances = zip(*sorted_features)
        print(f"\nGlobal explanations - LIME Feature Importance of Top {max_features} Features:")
        print("- This plot aggregates feature importance scores across multiple observations.")
        print("- Higher bars indicate features with a stronger average influence on predictions.")
        print("- The plot helps identify which features consistently contribute the most to the model's predictions.")



        fig, ax = plt.subplots(figsize=(10, 6))  
        ax.barh(features, importances, color="skyblue")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"LIME Feature Importance Summary (Top {max_features} Features)")
        ax.invert_yaxis()

        fig.tight_layout()

        if pdf:
            pdf.savefig(fig)

        plt.show()

        plt.close(fig)
    

    def top_features_by_lime_importance(self, explanations, X, top_n_non_binary=3, correlation_threshold=0.4):
        """
        Find the top N non-binary features (to display plots) and all non-binary features (to save plots to pdf)
        based on LIME Feature Importance, ensuring that selected features are not highly correlated with any other.

        Parameters:
            explanations: list, LIME explanation objects.
            X: pd.DataFrame, input features (used to compute correlations).
            top_n_non_binary: int, number of top non-binary features to select.
            correlation_threshold: float, maximum allowed correlation between selected features.

        Returns:
            tuple: (top_n_non_binary_features, top_m_all_features)
        """
        binary_features = [col for col in X.columns if X[col].nunique() == 2]

        feature_importances = {}
        for exp in explanations:
            for feature, weight in exp.as_list():
                normalized_feature = normalize_feature_name(feature)
                feature_importances[normalized_feature] = feature_importances.get(normalized_feature, 0) + abs(weight)

        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

        selected_top_non_binary = []
        selected_all_non_binary = []

        correlation_matrix = X.corr()

        for feature, _ in sorted_features:
            if feature not in binary_features and feature in correlation_matrix.columns:
                if is_uncorrelated(feature, selected_top_non_binary, correlation_matrix, correlation_threshold):
                    selected_top_non_binary.append(feature)
                    if len(selected_top_non_binary) >= top_n_non_binary:
                        break

        for feature, _ in sorted_features:
            if feature not in binary_features and feature in correlation_matrix.columns:
                if is_uncorrelated(feature, selected_all_non_binary, correlation_matrix, correlation_threshold):
                    selected_all_non_binary.append(feature)

        print(f"Top {len(selected_all_non_binary)} non-binary, uncorrelated features in terms of LIME importance:", selected_all_non_binary)

        return selected_top_non_binary, selected_all_non_binary