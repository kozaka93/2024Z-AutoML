from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

class PartialDependencePlotter:
    def __init__(self):
        """
        Initialize the PartialDependencePlotter class.
        """
        pass

    def generate_pdp(self, model, X, features_for_displaying_plots, features_for_saving_to_pdf, pdf=None):
        """
        Generate Partial Dependence Plots (PDPs) for specified features.

        This method creates PDPs to show the average effect of selected features on the model predictions. 
        PDPs are displayed for the features specified in `features_for_displaying_plots` and saved for all 
        features listed in `features_for_saving_to_pdf`.

        Parameters:
            model: object
                The trained machine learning model to explain.
            X: pd.DataFrame
                Input features used in the model.
            features_for_displaying_plots: list
                List of features for which the plots will be displayed.
            features_for_saving_to_pdf: list
                List of all features for which the plots will be generated and saved to the PDF.
            pdf: PdfPages or None, optional, default=None
                A PdfPages object to save plots. If None, plots are not saved.

        Workflow:
            1. Display PDPs for `features_for_displaying_plots`.
            2. Save PDPs for `features_for_saving_to_pdf` in batches of 4 plots per page to the PDF.

        """
        print("- Partial Dependence Plots (PDPs) show the average effect of one or more features on the predicted outcome.")
        print("- The x-axis represents the values of the selected feature(s), and the y-axis represents the predicted outcome.")
        print("- PDPs help identify patterns such as linearity, thresholds, or non-linear dependencies between features and predictions.")
        print("- These plots are generated only for non-binary features that are not strongly correlated with other features.")
        print("- Avoiding correlated features ensures the interpretations are not redundant or misleading.")
        print("- These plots are useful for understanding how a specific feature influences the model's decisions, holding other features constant.")

        print(f"PDP plots for all uncorrelated non-binary features will be saved to the PDF.")
        for feature in features_for_displaying_plots:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                model,
                X,
                [feature], 
                ax=ax
            )
            plt.title(f"Partial Dependence Plot for {feature}")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

        if pdf:
            for i in range(0, len(features_for_saving_to_pdf), 4):
                n_features = len(features_for_saving_to_pdf[i:i + 4]) 

                if n_features == 1: 
                    fig, ax = plt.subplots(figsize=(8, 6))
                    PartialDependenceDisplay.from_estimator(
                        model,
                        X,
                        [features_for_saving_to_pdf[i]],
                        ax=ax
                    )
                    ax.set_title(f"Partial Dependence Plot for {features_for_saving_to_pdf[i]}")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                else: 
                    rows = (n_features + 1) // 2  
                    fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows))
                    axes = axes.flatten() 

                    for j, feature in enumerate(features_for_saving_to_pdf[i:i + 4]):
                        PartialDependenceDisplay.from_estimator(
                            model,
                            X,
                            [feature], 
                            ax=axes[j]
                        )
                        axes[j].set_title(f"Partial Dependence Plot for {feature}")

                    for k in range(len(features_for_saving_to_pdf[i:i + 4]), len(axes)):
                        axes[k].set_visible(False)

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)