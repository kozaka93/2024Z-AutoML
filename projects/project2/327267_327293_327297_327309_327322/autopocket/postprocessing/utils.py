import re 


def validate_features_for_displaying(features, X):
        """
        Validate and filter the features_for_displaying_plots parameter.

        Parameters:
            features: list or None
                List of features to validate.
            X: pd.DataFrame
                DataFrame containing the feature columns.

        Returns:
            list or None: Validated list of non-binary feature names or None if the list is empty or invalid.
        """
        if features is not None:
            if not isinstance(features, list):
                raise ValueError("features_for_displaying_plots must be a list of strings.")

            if not all(isinstance(feature, str) for feature in features):
                raise ValueError("features_for_displaying_plots must contain only strings.")

            non_binary_columns = [col for col in X.columns if X[col].nunique() > 2]
            features = [feature for feature in features if feature in non_binary_columns]

            if not features:
                return None

        return features


def plot_features_with_explanations(best_model, X, explanations, lime_processor, pdp_plotter, ice_plotter, pdf, features_for_displaying_plots=None, subset_fraction_for_ICE_plot=None):
    """
    Generate and display Partial Dependence Plots (PDPs) and Individual Conditional Expectation (ICE) plots 
    for specified or automatically selected features based on LIME feature importance.

    Parameters:
        best_model: object
            The trained machine learning model to explain.
        X: pd.DataFrame
            Input features used in the model.
        explanations: list
            LIME explanations generated for the model.
        lime_processor: object
            Instance of LIME processor for handling feature importance and selection.
        pdp_plotter: object
            Instance of PDP plotter to generate Partial Dependence Plots.
        ice_plotter: object
            Instance of ICE plotter to generate Individual Conditional Expectation plots.
        pdf: PdfPages or None
            PDF file object to save plots. If None, plots will not be saved.
        features_for_displaying_plots: list, optional
            List of features to display plots for. If None, top features will be selected based on LIME importance.
        subset_fraction_for_ICE_plot: float, optional
            Fraction of rows from X to use for ICE plot generation. If None, use all rows.
    """
    if features_for_displaying_plots is None:
        print("Selecting top features based on LIME Feature Importance...")
        top_non_binary_features, all_non_binary_features = lime_processor.top_features_by_lime_importance(
            explanations=explanations,
            X=X,
            top_n_non_binary=3
        )
        print(f"Displaying Partial Dependence Plots for top {len(top_non_binary_features)} non-binary features...")
        pdp_plotter.generate_pdp(best_model, X, top_non_binary_features, all_non_binary_features, pdf=pdf)

        print(f"Displaying ICE plots for top {len(top_non_binary_features)} non-binary features...")
        ice_plotter.generate_ice(best_model, X, top_non_binary_features, all_non_binary_features, pdf=pdf, subset_fraction_for_ICE_plot=subset_fraction_for_ICE_plot)
    else:
        _, all_non_binary_features = lime_processor.top_features_by_lime_importance(
            explanations=explanations,
            X=X,
            top_n_non_binary=3
        )
        print(f"Displaying Partial Dependence Plots for {len(features_for_displaying_plots)} selected non-binary features...")
        pdp_plotter.generate_pdp(best_model, X, features_for_displaying_plots, all_non_binary_features, pdf=pdf)

        print(f"Displaying ICE plots for {len(features_for_displaying_plots)} selected non-binary features...")
        ice_plotter.generate_ice(best_model, X, features_for_displaying_plots, all_non_binary_features, pdf=pdf, subset_fraction_for_ICE_plot=subset_fraction_for_ICE_plot)


def is_uncorrelated(feature, selected_features, correlation_matrix, correlation_threshold=0.4):
    """
    Check if the feature is uncorrelated with the already selected features.
    """
    for selected_feature in selected_features:
        if abs(correlation_matrix.loc[feature, selected_feature]) > correlation_threshold:
            return False
    return True


def normalize_feature_name(feature_name):
    """
    Normalize feature names based on the total count of operators found.
    If two operators (e.g., '<' and '>') are found, take the string between them.
    If one operator is found, take the string to the left of it.
    Additionally, count occurrences of '<' and '>'.

    Parameters:
        feature_name: str, original feature name from LIME.

    Returns:
        tuple: (normalized_feature_name, total_operator_count)
    """
    operators = ["<", ">"]

    operator_counts = {op: feature_name.count(op) for op in operators}
    total_operator_count = sum(operator_counts.values())

    if total_operator_count >= 2:  
        start_op = operators[0] if operator_counts[operators[0]] > 0 else operators[1]
        end_op = operators[1] if operator_counts[operators[1]] > 0 else operators[0]

        pattern = rf"{re.escape(start_op)}(.*?) {re.escape(end_op)}"
        match = re.search(pattern, feature_name)
        if match:
            return match.group(1).strip()

    elif total_operator_count == 1:
        single_op = operators[0] if operator_counts[operators[0]] > 0 else operators[1]
        return feature_name.split(single_op)[0].strip()

    return feature_name.strip()

