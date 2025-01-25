from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class IndividualConditionalExpectationPlotter:
    def __init__(self):
        """
        Initialize the IndividualConditionalExpectationPlotter class.
        """
        pass
    

    def generate_ice(self, model, X, features_for_displaying_plots, features_for_saving_to_pdf, pdf=None, subset_fraction_for_ICE_plot=0.5):
        """
        Generate Individual Conditional Expectation (ICE) plots for features specified by parameter 
         'features_for_saving_to_pdf'.
        Display ICE for 'features_for_displaying_plots' and save all plots to the PDF.

        Parameters:
            model: The trained model to explain.
            X: pd.DataFrame, input features.
            features_for_displaying_plots: list, features for which the plots will be displayed.
            features_for_saving_to_pdf: list, all features for which the plots will be generated and saved to PDF.
            pdf: PdfPages object to save plots. If None, no plots are saved.
            subset_fraction_for_ICE_plot : float or None, optional. Fraction of rows from X to use for ICE plot generation. If None, use all rows.
        """

        if subset_fraction_for_ICE_plot is not None:
            if 0 < subset_fraction_for_ICE_plot <= 1:
                subset_fraction_for_ICE_plot = max(1, int(subset_fraction_for_ICE_plot * len(X)))
                X = X.sample(subset_fraction_for_ICE_plot, random_state=42)
                print(f"Using a sample of {subset_fraction_for_ICE_plot} observations from the input data.")
            else:
                raise ValueError("sample_size must be a float between 0 and 1.")

        print("- Individual Conditional Expectation (ICE) plots show how the predicted outcome changes when a single feature is varied.")
        print("- Unlike PDPs, ICE plots display individual trajectories for each observation, providing insight into heterogeneity in feature effects.")
        print("- The x-axis represents the values of the selected feature(s), and the y-axis represents the predicted outcome.")
        print("- ICE plots are generated only for non-binary features that are not strongly correlated with other features.")
        print("- This avoids redundancy and ensures clearer interpretations.")
        if subset_fraction_for_ICE_plot is not None:
            print(f"- Computational efficiency and plots readability is improved by sampling a fraction of the dataset - in this case plotted lines represents {subset_fraction_for_ICE_plot} observations.")
        else:
            print("- All observations are used for ICE plot generation.")
        print("- These plots are especially useful for identifying interactions and non-linear relationships in model predictions.")
        print(f"ICE plots for all uncorrelated non-binary features will be saved to the PDF.")
        for feature in features_for_displaying_plots:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                model,
                X,
                [feature],
                kind="both",
                ax=ax
            )
            min_val, max_val = ax.get_ylim()
            if min_val == max_val:
                delta = 0.1
                ax.set_ylim([min_val - delta, max_val + delta])

            plt.title(f"ICE Plot for {feature}")
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
                        kind="both",
                        ax=ax
                    )
                    min_val, max_val = ax.get_ylim()
                    if min_val == max_val:
                        delta = 0.1
                        ax.set_ylim([min_val - delta, max_val + delta])

                    ax.set_title(f"ICE Plot for {features_for_saving_to_pdf[i]}")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                else:  
                    rows = (n_features + 1) // 2 
                    fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows)) 
                    axes = axes.flatten() if n_features > 1 else [axes]

                    for j, feature in enumerate(features_for_saving_to_pdf[i:i + 4]):
                        PartialDependenceDisplay.from_estimator(
                            model,
                            X,
                            [feature],
                            kind="both",
                            ax=axes[j]
                        )
                        min_val, max_val = axes[j].get_ylim()
                        if min_val == max_val:
                            delta = 0.1
                            axes[j].set_ylim([min_val - delta, max_val + delta])

                        axes[j].set_title(f"ICE Plot for {feature}")

                    for k in range(len(features_for_saving_to_pdf[i:i + 4]), len(axes)):
                        axes[k].set_visible(False)

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

