from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import shap
import matplotlib.pyplot as plt


class PredictExplainer:
    def __init__(self, medaid, model):
        """
        Initializes the PredictExplainer class with the medaid object and the model.

        Parameters:
        - medaid: The medaid object that holds the necessary data or configurations.
        - model: The model used for making predictions.
        """
        self.medaid = medaid
        self.model = model
        self.preprocessing_details = pd.read_csv(self.medaid.path + '/results/preprocessing_details.csv')
        # Create dictionaries for encoders, scalers, and imputers
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

    def preprocess_input_data(self, input_data):
        """
        Preprocesses the input data using the stored preprocessing details.
        This version uses pandas get_dummies for one-hot encoding.
        """
        processed_data = input_data.copy()

        # Handle one-hot encoding using pandas get_dummies
        for _, row in self.preprocessing_details.iterrows():
            column_name = row["Column Name"]

            if column_name not in processed_data.columns:
                continue  # Skip columns not in the input data

            # Encoding handling: Apply encoding methods first
            if row["Encoded"] and not row["Removed"]:
                encoding_method = row["Encoding Method"]
                if encoding_method == "One-Hot Encoding":
                    # One-Hot encoding using pandas get_dummies
                    processed_data_encoded = pd.get_dummies(processed_data, columns=[column_name], drop_first=False)
                    # Ensure that columns in the input data match with the training data
                    processed_data_encoded = processed_data_encoded.reindex(columns=self.medaid.X.columns, fill_value=0)
                    processed_data = processed_data_encoded

                elif encoding_method == "Label Encoding":
                    if column_name not in self.encoders:
                        label_encoder = LabelEncoder()
                        if column_name in self.medaid.X.columns:
                            label_encoder.fit(self.medaid.X[column_name])
                        else:
                            label_encoder.fit(processed_data[column_name])
                        self.encoders[column_name] = label_encoder

                    # Transform the input data
                    processed_data[column_name] = self.encoders[column_name].transform(processed_data[column_name])

        # Now, handle imputation after encoding
        for _, row in self.preprocessing_details.iterrows():
            column_name = row["Column Name"]

            if column_name not in processed_data.columns:
                continue  # Skip columns not in the input data

            # Imputation handling
            if row["Imputation Method"] and not row["Removed"]:
                imputation_method = row["Imputation Method"]
                strategy = imputation_method.lower() if pd.notna(imputation_method) else "mean"

                if column_name not in self.imputers:
                    if processed_data[column_name].dtype in [np.float64, np.int64]:  # Numeric data
                        if strategy in ["mean", "median", "most_frequent"]:
                            imputer = SimpleImputer(strategy=strategy)
                        else:
                            raise ValueError(f"Unsupported imputation strategy for numeric data: {strategy}")
                    else:  # Categorical data
                        if strategy == "mean":
                            strategy = "most_frequent"  # Automatically change "mean" to "most_frequent" for categorical data
                        if strategy == "most_frequent":
                            imputer = SimpleImputer(strategy="most_frequent")
                        else:
                            raise ValueError(f"Unsupported imputation strategy for categorical data: {strategy}")

                    # Fit the imputer
                    if column_name in self.medaid.X.columns:
                        imputer.fit(self.medaid.X[[column_name]])
                    else:
                        imputer.fit(processed_data[[column_name]])

                    # Save the fitted imputer
                    self.imputers[column_name] = imputer

                # Transform the input data
                processed_data[column_name] = self.imputers[column_name].transform(
                    processed_data[column_name].values.reshape(-1, 1)
                ).flatten()

        # Scaling handling: Scale after encoding and imputation
        for _, row in self.preprocessing_details.iterrows():
            column_name = row["Column Name"]

            if column_name not in processed_data.columns:
                continue  # Skip columns not in the input data

            # Scaling handling
            if row["Scaling Method"] and not row["Removed"]:
                scaling_method = row["Scaling Method"].lower()

                if column_name not in self.scalers:
                    if scaling_method in ["standard scaling", "standardization"]:
                        scaler = StandardScaler()
                    elif scaling_method in ["min-max scaling", "normalization", "min_max"]:
                        scaler = MinMaxScaler()
                    else:
                        raise ValueError(f"Unsupported scaling method: {scaling_method}")

                    if column_name in self.medaid.X.columns:
                        scaler.fit(self.medaid.X[[column_name]])
                    else:
                        scaler.fit(processed_data[[column_name]])

                    self.scalers[column_name] = scaler

                # Transform the input data
                processed_data[column_name] = self.scalers[column_name].transform(
                    processed_data[column_name].values.reshape(-1, 1)
                ).flatten()

        return processed_data

    def _format_value(self, value):
        """
        Helper function to format values as integers or floating-point numbers with 3 decimal places.
        """
        if isinstance(value, float):
            return f"{value:.3f}"
        elif isinstance(value, int):
            return f"{value}"
        return value

    def analyze_prediction(self, prediction, target_column, prediction_proba):
        """
        Analyzes the predicted value of the target feature and compares it to the dataset.
        Incorporates SHAP feature importance plot in the classification report.
        """
        # Get target data
        df = self.medaid.df_before
        target_values = df[target_column]

        # Basic comparison with the classes distribution
        value_counts = target_values.value_counts(normalize=True) * 100

        # Retrieve y_labels dictionary
        y_labels = self.medaid.y_labels if self.medaid.y_labels else None

        # Generate path to SHAP feature importance plot
        shap_plot_path = f"shap_feature_importance/{self.model.__class__.__name__}_custom_feature_importance.png"

        # Initialize the analysis variable
        if len(value_counts) == 2:  # Binary classification
            analysis = f"""
            <div class="classification-report">
                <div class="report-header">
                    <h2>Classification Report</h2>
                </div>
                <div class="report-details">
                    <p><strong>Target Feature:</strong> {target_column}</p>
                    <p><strong>Predicted Value:</strong> {self._format_value(prediction)}</p>
                    <p><strong>Prediction Probability:</strong> {prediction_proba}</p>
                    <p><strong>Model Used:</strong> {self.model.__class__.__name__}</p>
                </div>
                <div class="report-analysis">
                    <h3>Prediction Analysis (Binary Classification):</h3>
                    <ul>
                        <li>Class 0 occurs in {value_counts.get(0, 0):.2f}% of patients.</li>
                        <li>Class 1 occurs in {value_counts.get(1, 0):.2f}% of patients.</li>
                        <li>The predicted class of {self._format_value(prediction)} is {'common' if value_counts.get(prediction, 0) > 50 else 'rare'} amongst other patients.</li>
                    </ul>
                </div>
                <div class="target-encoding">
                    <h3>Target Variable Encoding:</h3>
                    <ul>
            """
        else:  # Multiclass classification
            analysis = f"""
            <div class="classification-report">
                <div class="report-header">
                    <h2>Classification Report</h2>
                </div>
                <div class="report-details">
                    <p><strong>Target Feature:</strong> {target_column}</p>
                    <p><strong>Predicted Value:</strong> {self._format_value(prediction)}</p>
                    <p><strong>Prediction Probabilities:</strong> {prediction_proba}</p>
                    <p><strong>Model Used:</strong> {self.model.__class__.__name__}</p>
                </div>
                <div class="report-analysis">
                    <h3>Prediction Analysis (Multiclass Classification):</h3>
                    <ul>
                        <li>Class distribution:</li>
                        <ul>
            """
            for class_label, percentage in value_counts.items():
                analysis += f"<li>Class {class_label}: {percentage:.2f}% of patients.</li>"

            analysis += f"""
                        </ul>
                        <li>The predicted class of {self._format_value(prediction)} is {'common' if value_counts.get(prediction, 0) > 100 / len(value_counts) else 'rare'} amongst other patients.</li>
                    </ul>
                </div>
                <div class="target-encoding">
                    <h3>Target Variable Encoding:</h3>
                    <ul>
            """
        #If there is nothing in the y_labels dictionary, the target variable is not encoded, display that
        if not y_labels:
            analysis += f"<li>The target variable '{target_column}' is not encoded.</li>"
        # Add target variable encoding information
        for encoded_value, label in y_labels.items():
            analysis += f"<li>Encoded value {encoded_value}: {label}</li>"

        analysis += """
                    </ul>
                </div>
                <!--
                <div class="feature-importance">
                    <h3>Feature Importance on Whole Dataset:</h3>
                    <img src="{shap_plot_path}" alt="Feature Importance Plot for the whole dataset" style="max-width: 100%; height: auto; border: 1px solid #ccc; margin-top: 10px;">
                </div> -->
            </div>
            """

        return analysis

    def generate_html_report(self, df, input_data):
        """
        Generates an HTML report that compares the input data with the dataset.
        """
        # Classify and analyze features
        feature_analysis = self.classify_and_analyze_features(df, input_data)

        # Predict and analyze the target
        prediction, prediction_analysis, shap_force, shap_summary, lime_plot = self.predict_target(input_data)

        # Start HTML report
        html_report = f"""
        <!DOCTYPE html>
            <html lang='en'>
            <head>
                <meta charset='UTF-8'>
                 <style>
                   body {{
                       font-family: Arial, sans-serif;
                       background-color: #f4f4f4;
                       color: #333;
                   }}
                   .container {{
                       width: 80%;
                       margin: 0 auto;
                       background-color: white;
                       padding: 20px;
                       border-radius: 8px;
                       box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                   }}


                   .feature {{
                       margin-bottom: 20px;
                   }}
                   .feature-header {{
                       background-color: #003366;
                       color: white;
                       padding: 10px;
                       border-radius: 5px;
                       font-weight: bold;
                       cursor: pointer;
                   }}
                   .feature-content {{
                       display: none;
                       background-color: #f9f9f9;
                       padding: 10px;
                       border-left: 4px solid #003366;
                       border-radius: 5px;
                       margin-top: 10px;
                   }}
                   .feature-category {{
                       margin-top: 10px;
                   }}
                   .feature-value {{
                       font-weight: bold;
                       color: #003366;
                   }}
                   .btn {{
                       padding: 10px 20px;
                       background-color: #003366;
                       color: white;
                       border: none;
                       cursor: pointer;
                   }}
                   .btn:hover {{
                       background-color: #00509e;
                   }}
                   .prediction {{
                       margin-top: 30px;
                       background-color: #E1E9F1;
                       padding: 20px;
                       border-radius: 8px;
                   }}
                   .prediction-header {{
                       font-weight: bold;
                       color: #003366;
                       margin-bottom: 10px;
                   }}
                   .prediction-content {{
                       color: #333;
                   }}
                   .features-container {{
                       max-height: 500px;
                       overflow-y: scroll;
                       padding-right: 15px;
                   }}
                   .classification-report {{
                       font-family: Arial, sans-serif;
                       max-width: 600px;
                       margin: 20px auto;
                       border: 1px solid #ddd;
                       border-radius: 8px;
                       box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                       padding: 20px;
                       background-color: #f9f9f9;
                   }}


                   .report-header h2 {{
                       margin-top: 0;
                       font-size: 1.5em;
                       color: #333;
                       text-align: center;
                       border-bottom: 2px solid #3498db;
                       padding-bottom: 10px;
                   }}


                   .report-details {{
                       margin: 20px 0;
                   }}


                   .report-details p {{
                       margin: 8px 0;
                       font-size: 1em;
                       color: #555;
                   }}


                   .report-details p strong {{
                       color: #333;
                   }}


                   .report-analysis {{
                       margin-top: 20px;
                   }}


                   .report-analysis h3 {{
                       font-size: 1.2em;
                       color: #3498db;
                       margin-bottom: 10px;
                   }}


                   .report-analysis ul {{
                       list-style-type: disc;
                       margin: 10px 0 0 20px;
                       color: #555;
                   }}


                   .report-analysis ul ul {{
                       list-style-type: circle;
                       margin-left: 20px;
                   }}


                   .report-analysis li {{
                       margin: 5px 0;
                   }}


                   /* SHAP Visualizations Section */
                   .shap-visualization {{
                       margin-top: 40px;
                       background-color: #f4f9fc;
                       padding: 20px;
                       border-radius: 8px;
                       box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                   }}
                   .shap-visualization h3 {{
                       font-size: 1.2em;
                       color: #003366;
                       margin-bottom: 10px;
                   }}
                   .shap-visualization iframe {{
                       width: 100%;
                       height: 350px;  /* Reduced height for Force Plot */
                       border: none;
                       border-radius: 8px;
                   }}
                   .shap-visualization img {{
                       display: block;
                       margin: 0 auto;  /* Center the image */
                       width: auto;  /* Maintain aspect ratio */
                       max-width: 80%;  /* Limit width to 80% of the page */
                       border-radius: 8px;
                       margin-top: 20px;
                   }}


                   .interpretation {{
                       margin-top: 20px;
                       font-size: 1em;
                       color: #555;
                   }}
                   .interpretation h4 {{
                       font-size: 1.1em;
                       color: #3498db;
                       margin-bottom: 10px;
                   }}
                   .interpretation p {{
                       margin: 8px 0;
                       color: #555;
                   }}
               </style>
               <script>
                   function toggleFeature(id) {{
                       var content = document.getElementById(id);
                       if (content.style.display === "none") {{
                           content.style.display = "block";
                       }} else {{
                           content.style.display = "none";
                       }}
                   }}
               </script>
            </head>
            <body>
                <div class="container">
                    <h1>Patient Data and Prediction Report</h1>

                    <!-- Prediction Section -->
                    <div class="prediction">
                        <div class="prediction-content">
                            {prediction_analysis}
                        </div>
                    </div>

                    <!-- Feature Analysis Section -->
                    {feature_analysis}

                    <!-- Visualizations Section -->
                    <div class="visualizations">
                        <h3>Model Interpretability Visualizations</h3>
        """

        # Add SHAP Force Plot if available
        if shap_force:
            html_report += f"""
                <div class="shap-visualization">
                    <h4>SHAP Force Plot:</h4>
                    <iframe src="shap_force_plot.html" frameborder="0"></iframe>
                    <!-- Interpretation for Force Plot -->
                       <div class="interpretation">
                           <h4>How to Interpret the Force Plot:</h4>
                           <p>
                               The force plot shows how each feature in the patient's data pushes the prediction either towards or away from the predicted class. The length of each arrow indicates the magnitude of the effect of the corresponding feature, while the color shows whether the feature is pushing the prediction in a positive (towards the class) or negative (away from the class) direction. The baseline (middle) represents the average prediction, and the arrows reflect how the features of the individual patient's data influence that prediction.
                           </p>
                       </div>

                </div>
            """

        # Add SHAP Summary Plot if available
        if shap_summary:
            html_report += f"""
                <div class="shap-visualization">
                    <h4>SHAP Summary Plot:</h4>
                    <img src="shap_summary_plot.png" alt="SHAP Summary Plot">
                    <!-- Interpretation for Summary Plot -->
                       <div class="interpretation">
                           <h4>How to Interpret the Summary Plot:</h4>
                           <p>
                               The summary plot shows the overall impact of each feature across the entire dataset. Each point represents a SHAP value for an individual prediction, and the features are sorted by their average impact on the model's output. The color represents the feature value, with red indicating higher values and blue indicating lower values. The spread of each feature's points gives an indication of the variation in feature impact across all samples in the dataset.
                           </p>
                       </div>

                </div>
                
            """

        # Add LIME Plot if available
        if lime_plot:
            html_report += f"""
                <div class="lime-visualization">
                    <h4>LIME Explanation:</h4>
                    <iframe src="lime_explanation.html" frameborder="0" width="100%" height="400px"></iframe>
                    <!-- Interpretation for LIME Explanation -->
            <div class="interpretation">
                <h4>How to Interpret the LIME Explanation:</h4>
                <p>
                    The LIME plot provides a local explanation for a specific prediction made by the model. It shows how the individual features of the patient's data contribute to the model's prediction. Each feature is assigned a weight that represents its impact on the predicted outcome. Features with higher weights have a more significant influence on the prediction. The plot typically visualizes the contributions of each feature, with the color representing whether the feature is pushing the model's prediction in a positive or negative direction.
                </p>
            </div>
                </div>
            """

        html_report += """
                </div>
            </body>
        </html>
        """
        return html_report

    def generate_viz(self, input_data):
        """
        Generates visualizations for the given input data using SHAP or LIME.
        """
        input_data = self.preprocess_input_data(input_data)
        if isinstance(self.model, (DecisionTreeClassifier, RandomForestClassifier)):
            # Use LIME for tree-based models
            return self.generate_lime_viz(input_data)
        else:
            # Use SHAP for other models
            return self.generate_shap_viz(input_data)

    def generate_shap_viz(self, input_data):
        """
        Generates SHAP visualizations for the given input data.

        This function uses the SHAP `Explanation` object to create a force plot
        for a single prediction and a summary plot for the whole dataset. Both are saved as files.
        """
        # Preprocess input data
        processed_input_data = self.preprocess_input_data(input_data)

        # Determine the SHAP explainer based on the model type
        if isinstance(self.model, (DecisionTreeClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.Explainer(self.model, self.medaid.X)

        # Generate SHAP explanations (returns a shap.Explanation object)
        explanation = explainer(processed_input_data)
        explanation_full = explainer(self.medaid.X)

        # Ensure features are 2-dimensional
        features = explanation[0].data
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)

        # Handle multiclass SHAP values
        if len(explanation[0].values.shape) == 2:
            # Multiclass case: values are (num_classes, num_features)
            class_index = 0  # Select the first class, or customize based on the desired class
            shap_values = explanation[0].values[class_index]
            base_value = explanation[0].base_values[class_index]
        else:
            # Binary classification or regression case
            shap_values = explanation[0].values
            base_value = explanation[0].base_values

        # Generate the force plot
        force_plot_path = f"{self.medaid.path}/shap_force_plot.html"
        force_plot = shap.plots.force(
            base_value,
            shap_values,
            features[0],  # Single instance, first sample
            feature_names=explanation.feature_names,
            matplotlib=False
        )
        shap.save_html(force_plot_path, force_plot)

        # Handle multiclass SHAP values
        if len(explanation_full.values.shape) == 3:  # (num_samples, num_classes, num_features)
            class_index = 0  # Choose a specific class (adjust as needed)
            shap_values = explanation_full.values[:, class_index, :]  # Extract SHAP values for the chosen class
        else:
            shap_values = explanation_full.values  # For binary classification or regression

        # Generate the SHAP summary plot
        summary_plot_path = f"{self.medaid.path}/shap_summary_plot.png"
        shap.summary_plot(
            shap_values,  # SHAP values for the selected class
            self.medaid.X,  # Input data (features)
            feature_names=explanation_full.feature_names,  # Feature names for labeling
            show=False
        )
        plt.savefig(summary_plot_path)

        # Return paths to the saved plots
        return {
            'force_plot': force_plot_path,
            'summary_plot': summary_plot_path
        }

    def generate_lime_viz(self, input_data):
        """
        Generates LIME visualizations for the given input data.
        """
        explainer = LimeTabularExplainer(
            training_data=self.medaid.X.values,  # Training data
            feature_names=self.medaid.X.columns.tolist(),  # Feature names
            class_names=[str(c) for c in self.model.classes_],  # Class labels
            mode='classification'  # Classification mode
        )

        # Generate explanation for the first prediction
        exp = explainer.explain_instance(
            data_row=input_data.iloc[0].values,  # Single row of input data
            predict_fn=self.model.predict_proba  # Prediction function
        )

        lime_plot_path = f"{self.medaid.path}/lime_explanation.html"
        exp.save_to_file(lime_plot_path)

        return {'lime_plot': lime_plot_path}

    def predict_target(self, input_data):
        """
        Preprocesses the input data and predicts the target feature using the model.
        Adds SHAP or LIME visualizations for enhanced interpretability.
        """
        # Preprocess the input data
        processed_input_data = self.preprocess_input_data(input_data)

        # Make and analyze the prediction
        prediction = self.model.predict(processed_input_data)[0]
        prediction_proba = self.model.predict_proba(processed_input_data)[0]
        target_column = self.medaid.target_column
        prediction_analysis = self.analyze_prediction(prediction, target_column, prediction_proba)

        # Generate visualizations (SHAP or LIME)
        viz = self.generate_viz(input_data)

        # Extract visualization paths based on whether SHAP or LIME was used
        shap_force = viz.get('force_plot')  # For SHAP Force Plot
        shap_summary = viz.get('summary_plot')  # For SHAP Summary Plot
        lime_plot = viz.get('lime_plot')  # For LIME Plot

        return prediction, prediction_analysis, shap_force, shap_summary, lime_plot

    def classify_and_analyze_features(self, df, input_data):
        """
        Classifies features into categories (binary, categorical_strings, categorical_numbers, numerical_continuous)
        and analyzes each one based on its type.
        """
        feature_analysis = ""  # Holds the HTML for all features

        for column in df.columns:
            feature_content = ""
            # Check if the column is categorical (i.e., contains strings)
            if df[column].dtype == 'object':
                feature_content = self._analyze_categorical_strings(df, column, input_data[column])
            else:
                # Determine if the column is binary, categorical numeric, or continuous numeric
                unique_values = df[column].nunique()

                if unique_values == 2:  # Binary feature (usually encoded as 0 and 1)
                    feature_content = self._analyze_binary(df, column, input_data[column])
                elif 2 < unique_values < 10 and df[column].dtype in ['int64', 'float64']:  # Categorical numbers
                    feature_content = self._analyze_categorical_numbers(df, column, input_data[column])
                elif df[column].dtype in ['int64', 'float64']:  # Numerical continuous (e.g., age, BMI)
                    feature_content = self._analyze_numerical_continuous(df, column, input_data[column])

            feature_analysis += f"<div class='feature'>{feature_content}</div>"

        return f"<div class='features-container'>{feature_analysis}</div>"

    def _analyze_binary(self, df, column, input_value):
        """
        Generates HTML for a binary feature with collapsible content.
        """
        input_value = input_value.iloc[0]  # Ensure scalar
        value_counts = df[column].value_counts(normalize=True) * 100
        return f"""
            <div class="feature-header" onclick="toggleFeature('{column}_binary')">Feature '{column}' - Value: {input_value}</div>
            <div class="feature-content" id="{column}_binary">
                The new patient has a value of <span class="feature-value">{input_value}</span>.
                <div class="feature-category">This value occurs in <span class="feature-value">{value_counts[input_value]:.3f}%</span> of other patients.</div>
            </div>
        """

    def _analyze_categorical_numbers(self, df, column, input_value):
        """
        Generates HTML for a categorical numeric feature with collapsible content.
        """
        input_value = input_value.iloc[0]  # Ensure scalar
        value_counts = df[column].value_counts(normalize=True) * 100
        categories = [f"Value '{value}' occurs in {count:.3f}% of patients." for value, count in value_counts.items()]

        return f"""
            <div class="feature-header" onclick="toggleFeature('{column}_categorical_numbers')">Feature '{column}' - Value: {input_value}</div>
            <div class="feature-content" id="{column}_categorical_numbers">
                The new patient has a value of <span class="feature-value">{input_value}</span>.
                <div class="feature-category">
                    {'This value is rare amongst other patients.' if input_value not in value_counts.index else f'This value occurs in <span class="feature-value">{value_counts[input_value]:.3f}%</span> of patients.'}
                </div>
                <div class="feature-list">
                    <strong>All possible categories and their frequencies:</strong>
                    <ul>
                        {''.join([f"<li>{cat}</li>" for cat in categories])}
                    </ul>
                </div>
            </div>
        """

    def _analyze_categorical_strings(self, df, column, input_value):
        """
        Generates HTML for a categorical string feature with collapsible content.
        """
        input_value = input_value.iloc[0]  # Ensure scalar
        value_counts = df[column].value_counts(normalize=True) * 100

        return f"""
            <div class="feature-header" onclick="toggleFeature('{column}_categorical_strings')">Feature '{column}' - Value: '{input_value}'</div>
            <div class="feature-content" id="{column}_categorical_strings">
                The new patient has a value of <span class="feature-value">'{input_value}'</span>.
                <div class="feature-category">
                    {'This value is rare among ther patients.' if input_value not in value_counts.index else f'This value occurs in <span class="feature-value">{value_counts[input_value]:.3f}%</span> of patients.'}
                </div>
            </div>
        """

    def _analyze_numerical_continuous(self, df, column, input_value):
        """
        Generates HTML for a continuous numerical feature with collapsible content.
        """

        mean = df[column].mean()
        median = df[column].median()
        std_dev = df[column].std()
        min_value = df[column].min()
        max_value = df[column].max()


        input_value = input_value.iloc[0]  # Ensure scalar
        if input_value > mean + std_dev:
            comparison = "significantly above"
        elif input_value > mean:
            comparison = "slightly above"
        elif input_value == mean:
            comparison = "equal to"
        elif input_value < mean - std_dev:
            comparison = "significantly below"
        else:
            comparison = "slightly below"

        return f"""
            <div class="feature-header" onclick="toggleFeature('{column}_numerical_continuous')">Feature '{column}' - Value: {input_value:.3f}</div>
            <div class="feature-content" id="{column}_numerical_continuous">
                The new patient has a value of <span class="feature-value">{input_value:.3f}</span>. This value is {comparison} the mean value of <span class="feature-value">{mean:.3f}</span> for other patients.
                <div class="feature-category">
                    <strong>Additional details:</strong>
                    <ul>
                        <li>Median value: {median:.3f}</li>
                        <li>Standard deviation: {std_dev:.3f}</li>
                        <li>Min: {min_value:.3f}</li>
                        <li>Max: {max_value:.3f}</li>
                    </ul>
                </div>
            </div>
        """

"""
if __name__ == "__main__": #main stworzony do celów testowych
    # Load the medaid object
    with open('medaid1/medaid.pkl', 'rb') as file:
        medaid = pickle.load(file)

    model = medaid.best_models[1]
    print(model.__class__.__name__)
    pe = PredictExplainer(medaid, model)

    # Prepare the input data
    df = medaid.df_before.drop(columns=[medaid.target_column])
    input_data = medaid.df_before.head(1).drop(columns=[medaid.target_column])

    # Generate the HTML report
    html_report = pe.generate_html_report(df, input_data)
    with open('report_predict_and_features.html', 'w') as f:
        f.write(html_report) """
