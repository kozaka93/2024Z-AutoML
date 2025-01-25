from pylatex import Command, Document, Section, Subsection, Tabular
from pylatex.utils import NoEscape
from pylatex.table import Table 
import pandas as pd
import io, os
from pylatex import Section, Subsection, Figure, NoEscape, Subsubsection
from .plot_generator import PlotGenerator
import joblib
from ..xai.explain_decision_tree import ExplainDecisionTree
from ..xai.explain_random_forest import ExplainRandomForest
from ..xai.explain_xgboost import ExplainXGBoost

class ReportGenerator:
    def __init__(self, dataset, dataset_name, optimizer):
        """
        this class is responsible for generating the report in the pdf format
        """
        self.doc = Document()
        self.dataset = pd.DataFrame(dataset)
        self.dataset_name = dataset_name
        # save optimizer instance, to get hyperparameters and metrics
        self.optimizer = optimizer

        # delete clf__ from the column names
        self.optimizer.params_rf.columns = [col.replace('clf__', '') for col in self.optimizer.params_rf.columns]
        self.optimizer.params_dt.columns = [col.replace('clf__', '') for col in self.optimizer.params_dt.columns]
        self.optimizer.params_xgb.columns = [col.replace('clf__', '') for col in self.optimizer.params_xgb.columns]

        # get metrics for each model
        self.optimizer.metrics_xgb = self.optimizer.params_xgb[['f1', 'accuracy', 'roc_auc']]
        self.optimizer.metrics_xgb['model'] = 'XGBoost'
        self.optimizer.metrics_rf = self.optimizer.params_rf[['f1', 'accuracy', 'roc_auc']]
        self.optimizer.metrics_rf['model'] = 'Random Forest'
        self.optimizer.metrics_dt = self.optimizer.params_dt[['f1', 'accuracy', 'roc_auc']]
        self.optimizer.metrics_dt['model'] = 'Decision Tree'
        
        # concatenate metrics
        self.metrics = pd.concat([self.optimizer.metrics_xgb, self.optimizer.metrics_rf, self.optimizer.metrics_dt])

        # create Models folder
        os.makedirs(f'Results/{self.dataset_name}/Models', exist_ok=True)
        
        # save best models
        joblib.dump(self.optimizer.best_xgb_instance, f'Results/{self.dataset_name}/Models/best_XGBoost.joblib')
        joblib.dump(self.optimizer.best_rf_instance, f'Results/{self.dataset_name}/Models/best_RandomForest.joblib')
        joblib.dump(self.optimizer.best_dt_instance, f'Results/{self.dataset_name}/Models/best_DecisionTree.joblib')

        # create explainer instances for the best models
        self.explainer_best_xgb = ExplainXGBoost(self.optimizer.best_xgb_instance)
        self.explainer_best_rf = ExplainRandomForest(self.optimizer.best_rf_instance)
        self.explainer_best_dt = ExplainDecisionTree(self.optimizer.best_dt_instance)


    def make_small_margins(self):
        '''
        This method reduces the margins of the document to make it more compact
        '''
        self.doc.packages.append(Command('usepackage', 'geometry'))
        self.doc.packages.append(Command('geometry', 'margin=0.2in'))


    def new_page(self):
        '''
        This method adds a new page to the report
        '''
        self.doc.append(NoEscape(r'\newpage'))


    def add_title(self):
        '''
        This method adds a title to the report
        '''
        self.doc.preamble.append(Command('title', "Report on " + self.dataset_name + " dataset"))
        self.doc.preamble.append(Command('author', 'Classify2TeX'))
        self.doc.append(NoEscape(r'\maketitle'))
        self.new_page()   


    def add_table_of_contents(self):
        '''
        This method adds a table of contents to the report
        '''
        self.doc.append(NoEscape(r'\tableofcontents'))
        self.new_page()


    def add_info_table(self):
        '''
        This method adds a table with the dataset information (using .info())
        '''
        # Capture dataset.info() into a buffer
        buffer = io.StringIO()
        self.dataset.info(buf=buffer)
        dataset_info = buffer.getvalue()

        # Extract parts of dataset info() for table formatting
        lines = dataset_info.split('\n')
        table_rows = []

        # Process lines to extract table header and rows
        for line in lines:
            if "Non-Null Count" in line and "Dtype" in line:
                header = ["Column", "Non-Null Count", "Dtype"]
            elif line.strip() and line.strip()[0].isdigit():
                parts = line.split()
                column_name = parts[1]
                non_null_count = parts[2]
                dtype = parts[-1]
                table_rows.append([column_name, non_null_count, dtype])

        # Convert extracted information to a dataframe
        info_df = pd.DataFrame(table_rows, columns=header)

        # Use the print_dataframe method to print the dataframe
        self.print_dataframe(
            df=info_df,
            caption='Dataset Columns Information',
            num_after_dot=0
        )

    def add_describe_info(self):
        '''
        This method adds a table with the dataset's descriptive statistics (using .describe())
        '''
        # Calculate descriptive statistics and reset the index to include row labels
        describe_data = self.dataset.describe().transpose().reset_index()
        describe_data.rename(columns={'index': 'Column Name/Statistic'}, inplace=True)

        # Use the print_dataframe method to print the dataframe
        self.print_dataframe(
            df=describe_data,
            caption='Dataset Descriptive Statistics',
            num_after_dot=2
        )

    def add_bar_charts(self):
        '''
        This method adds bar charts for each categorical column in the dataset
        '''
        self.new_page()

        plt = PlotGenerator().generate_bar_charts(self.dataset, self.dataset_name)
        if plt is None:
            return
        
        with self.doc.create(Subsubsection('Bar Charts of Categorical columns')):
            self.doc.append(NoEscape(r'The bar charts below show the distribution of categorical features in the dataset.'))
            # put image in the latex document
            with self.doc.create(Figure(position='h!')) as fig:
                fig.add_image(f'EDA/bar_charts.png', width='460px')
                fig.add_caption('Bar Charts of Categorical columns')


    def add_histograms(self):
        '''
        This method adds histograms for each numerical column in the dataset
        '''
        plt = PlotGenerator().generate_histograms(self.dataset, self.dataset_name)
        if plt is None:
            return
        
        with self.doc.create(Subsubsection('Histograms of Numerical columns')):
            self.doc.append(NoEscape(r'The histograms below show the distribution of numerical features in the dataset.'))
            # put image in the latex document
            with self.doc.create(Figure(position='h!')) as fig:
                fig.add_image(f'EDA/histograms.png', width='460px')
                fig.add_caption('Histograms of Numerical columns')
        return 

        
    def add_metrics_description(self):
        """
        Adds a section describing evaluation metrics to the LaTeX document.
        """
        # Add required packages to the preamble
        self.doc.preamble.append(NoEscape(r'\usepackage{amsmath}'))
        self.doc.preamble.append(NoEscape(r'\usepackage{amssymb}'))
        self.doc.preamble.append(NoEscape(r'\usepackage{enumitem}'))

        # Add the Evaluation Metrics section
        with self.doc.create(Section('Evaluation Metrics')):
            
            # Accuracy subsection
            with self.doc.create(Subsection('Accuracy')):
                self.doc.append(NoEscape(r"""
                \textbf{Accuracy} is one of the simplest evaluation metrics for classification models. 
                It is defined as the ratio of correctly predicted observations to the total number of observations:

                \[
                \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
                \]

                While accuracy is intuitive and easy to understand, it may not be suitable for imbalanced datasets. 
                For example, in a dataset where 95\% of the samples belong to one class, predicting the majority class for every instance 
                would result in high accuracy but poor performance on the minority class.
                """))

            # F1 Score subsection
            with self.doc.create(Subsection('F1 Score')):
                self.doc.append(NoEscape(r"""
                The \textbf{F1 Score} is the harmonic mean of Precision and Recall, providing a balance between the two. 
                It is particularly useful when dealing with imbalanced datasets. Precision and Recall are defined as follows:

                \[
                \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
                \]
                \[
                \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
                \]

                The F1 Score combines these metrics:

                \[
                \text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
                \]

                A high F1 Score indicates a good balance between Precision and Recall, making it a valuable metric in scenarios where false positives 
                and false negatives have significant costs.
                """))

            # ROC AUC subsection
            with self.doc.create(Subsection('ROC AUC')):
                self.doc.append(NoEscape(r"""
                The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings. 
                The \textbf{Area Under the Curve (AUC) of the ROC curve} measures the overall ability of the model to distinguish between classes. 

                \[
                \text{AUC} = \int_{\text{FPR}=0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})
                \]

                Key points about ROC AUC:
                \begin{itemize}
                    \item An AUC of 0.5 indicates random guessing.
                    \item An AUC of 1.0 indicates perfect classification.
                    \item It is a threshold-independent metric, providing an aggregate measure of performance across all classification thresholds.
                \end{itemize}

                ROC AUC is particularly useful for binary classification tasks and provides insights into the trade-off between sensitivity and specificity.
                """))

    def add_shap_description(self):
        """
        This method adds a section describing SHAP values to the LaTeX document.
        """
        self.doc.append(NoEscape(r"""
        \textbf{SHAP (SHapley Additive exPlanations) values} are a unified measure of feature importance, grounded in cooperative game theory, that explain the contribution of each feature to the predictions of a machine learning model. By assigning a consistent and fair contribution to each feature, SHAP values offer insights into the underlying decision-making process of the model, both for specific predictions and overall feature importance.
        """))

        self.doc.append(NoEscape(r"""
        The fundamental principle behind SHAP is that a model’s prediction for a given instance can be decomposed into the sum of contributions from its features, along with a baseline value. The baseline typically represents the average model prediction across the dataset when no feature information is provided.
        """))

        self.doc.append(NoEscape(r"""
        \subsection*{How SHAP Values work}
        """))

        self.doc.append(NoEscape(r"""
        For a specific instance, SHAP calculates how much each feature contributes to the difference between the baseline and the model's prediction. This involves:
        """))

        self.doc.append(NoEscape(r"""
        \begin{enumerate}
            \item \textbf{Marginal Contributions}: Evaluating how the prediction changes when each feature is added to subsets of other features. For example, if you have features \( A, B, C \), SHAP will compute how the prediction changes when \( A \) is added to subsets like \( \{\}, \{ B \}, \{ C \}, \{ B, C \} \), etc.
            \item \textbf{Weighted Averaging Across Subsets}: To compute the SHAP value for a feature, the method takes the average of its marginal contributions across all subsets of features, weighted by the size of the subsets. This ensures fairness in the distribution of contributions.
            \item \textbf{Baseline Value}: The baseline is a reference point, usually the average model prediction over the dataset. It represents what the model predicts when no features are considered.
        \end{enumerate}
        """))

        self.doc.append(NoEscape(r"""
        For any given data point, SHAP values indicate how much each feature shifts the model’s prediction relative to the baseline. A positive SHAP value means the feature increases the prediction, while a negative SHAP value means it decreases the prediction. This decomposition allows for a granular understanding of both the direction and magnitude of each feature's influence on the model’s decision."""))

    def add_shap_bar_plot_description(self):
        """
        This method adds a description of the SHAP bar plot to the LaTeX document.
        """
        self.doc.append(NoEscape(r"""
        \textbf{SHAP bar plot} provides a concise overview of the importance of individual features in the model's predictions. 
        Each bar represents a feature, with its length corresponding to the mean absolute SHAP value across all samples. 
        This indicates the average magnitude of the feature's contribution to the predictions, regardless of direction.
        """))

        self.doc.append(NoEscape(r"""
        \hspace{20px}Features are ranked in descending order of importance, and only the top 15 features are displayed by default for clarity.
        The bar plot allows quick identification of the most influential features driving the model's behavior and is particularly useful for comparing their relative contributions.\\
        \newline                     
        """))

    def add_violin_plot_description(self):
        """
        This method adds a description of the SHAP violin plot to the LaTeX document.
        """
        self.doc.append(NoEscape(r"""
        \textbf{SHAP violin plot} provides a visual summary of how each feature influences model predictions and the variability of this influence. 
        Features are listed on the vertical axis in descending order of importance, while the horizontal axis shows SHAP values, 
        indicating the magnitude and direction of each feature's contribution to predictions.
        """))

        self.doc.append(NoEscape(r"""
        \hspace{20px}The shape of each 'violin' represents the distribution of SHAP values for a feature: wider sections indicate higher density of similar values, 
        while narrower sections show less frequent SHAP values. Positive SHAP values increase the prediction, and negative values decrease it.
        """))

        self.doc.append(NoEscape(r"""
        \hspace{20px} Colors correspond to actual feature values, with red typically representing higher values and blue lower ones.
        The color distribution along the SHAP scale highlights how feature values affect predictions; for instance, 
        if red dominates the positive side, high feature values increase predictions.
        """))

        self.doc.append(NoEscape(r"""
        \hspace{10px} By default, only the top 15 features by importance are displayed, keeping the visualization focused and interpretable.\\
        """))

    def add_feature_importance_plot_description(self):
        """
        This method adds a description of the feature importance bar chart to the LaTeX document.
        """
        self.doc.append(NoEscape(r"""
        \textbf{Feature importance bar chart:} visually represents the contributions of individual features to the model's predictions, 
        based on their calculated importance scores. Each bar in the chart corresponds to a feature, and its length indicates the magnitude of 
        that feature's importance in reducing split impurity during the model's training process. Features that contribute more significantly to 
        the model's predictive accuracy are displayed with longer bars, while less influential features have shorter bars.
        """))

        self.doc.append(NoEscape(r"""
        \hspace{20px}The chart is horizontally oriented, with feature names listed on the vertical axis and their corresponding importance values on the horizontal axis.
        """))

        self.doc.append(NoEscape(r"""
        \hspace{20px}This visualization is especially useful for diagnosing the model's behavior, understanding which features drive its decisions, 
        and identifying variables that have the most impact on predictions.\\
        \newline
        """))

    def print_dataframe(self, df, caption, num_after_dot=2, no_index=False):
        '''
        This method prints the entire dataframe to the report.

        Args:
            df: The dataframe to be printed in the report.
            caption: The caption for the table in the report.
            num_after_dot: The number of decimal places to round numerical values in the dataframe.
            no_index: If True, the index column will not be included in the table.
        '''
        # Create the table with a caption
        with self.doc.create(Table(position='h!')) as table:
            table.add_caption(caption)

            self.doc.append(NoEscape(r'\vspace{0.2cm}')) # Add vertical space
            self.doc.append(NoEscape(r'\centering'))  # Center the table

            # Extract header and rows from the dataframe
            header = list(df.columns)
            table_rows = df.values.tolist()

            # Round numerical values to the specified number of decimal places
            rounded_rows = []
            for row in table_rows:
                rounded_row = [
                    round(value, num_after_dot) if isinstance(value, (float, int)) else value
                    for value in row
                ]
                rounded_rows.append(rounded_row)

            # Adjust the number of columns to match the dataframe
            with table.create(Tabular('|c|' * (len(header) + 1))) as tabular:
                tabular.add_hline()
                if no_index==False:
                    tabular.add_row(['Index'] + header)  # Add index column
                tabular.add_hline()

                # Add all rows to the table
                for i, row in enumerate(rounded_rows):
                    tabular.add_row([i] + row)

                tabular.add_hline()
                        

    def generate_report(self):
        '''
        This method generates the report, using the methods defined above.
        '''
        self.make_small_margins()  # Reduce the margins of the document
        self.add_title()  # Add the title to the report
        self.add_table_of_contents()  # Add table of contents

        with self.doc.create(Section('Exploratory Data Analysis')):

            with self.doc.create(Subsection('Non-Null Count, Dtype of features')):
                self.doc.append(NoEscape(r'The table 1 provides information about the dataset, including the number of non-null values and the data types of each feature.'))
                self.add_info_table()  # Add dataset info() table

            self.new_page()

            with self.doc.create(Subsection('Descriptive Statistics')):
                self.doc.append(NoEscape(r'The table 2 provides descriptive statistics for the dataset, including the count, mean, standard deviation, minimum, and maximum values.'))
                self.add_describe_info()  # Add dataset describe() table
            
            self.new_page()
            with self.doc.create(Subsection('Distribution of features')):
                self.doc.append(NoEscape(r'This section provides a visual representation of the distribution of features in the dataset using histograms (numerical features) and bar charts (categorical features). These visualizations can help in understanding the data.'))
                self.add_histograms() # Add histograms for numerical columns
                self.add_bar_charts() # Add bar charts for categorical columns

        self.new_page()
        self.add_metrics_description()

        self.new_page()
        with self.doc.create(Section('Model Optimization Results')):
            with self.doc.create(Subsection('Optimization Results Tables')):
                self.doc.append(NoEscape(r'The tables below show the hyperparameters and achieved metrics for each model configuration considered during the optimization process. The index of models with default hyperparameters is 0. The next models, indexed from 1, were chosen by Random Search.'))
                self.print_dataframe(self.optimizer.params_rf.transpose().reset_index().rename(columns={"index": "Metric/Hyperp.\ Iteration"}), 'Random Forest Hyperparameters and achivied metrics', num_after_dot=4)
                self.print_dataframe(self.optimizer.params_dt.transpose().reset_index().rename(columns={"index": "Metric/Hyperp. \ Iteration"}), 'Decision Tree Hyperparameters and achivied metrics', num_after_dot=4)
                self.print_dataframe(self.optimizer.params_xgb.transpose().reset_index().rename(columns={"index": "Metric/Hyperp. \ Iteration"}), 'XGBoost Hyperparameters and achivied metrics', num_after_dot=4)

            self.new_page()
            with self.doc.create(Subsection('Boxplots of accuracy, f1, roc_auc')):
                self.doc.append(NoEscape(r'Boxplots of accuracy, F1, and ROC AUC illustrate the distribution and variability of model performance metrics across different configurations of hyperparameters. The plots are located below.'))
                plt = PlotGenerator().generate_box_plots_metrics(self.metrics, self.dataset_name)
                # put image in the latex document
                with self.doc.create(Figure(position='h!')) as fig:
                    fig.add_image(f'ModelOptimization/box_plots_metrics.png', width='460px')
                    fig.add_caption('Boxplots of accuracy, f1, roc_auc')

            # add barplots of maximum values of metrics
            with self.doc.create(Subsection('Barplots of maximum values of metrics achievied by model')):
                self.doc.append(NoEscape(r'Barplots of maximum metric values show the highest performance scores for each model type. The plots are located below.'))
                plt = PlotGenerator().generate_barplots_max_metric(self.metrics, self.dataset_name)
                # put image in the latex document
                with self.doc.create(Figure(position='h!')) as fig:
                    fig.add_image(f'ModelOptimization/barplots_max_metric.png', width='460px')
                    fig.add_caption('Barplots of maximum values of metrics achievied by model')

                    # save to folder results of hyperparameters optimization
                    self.optimizer.params_xgb.to_csv(f'Results/{self.dataset_name}/ModelOptimization/XGBoost_hyperparameters_metrics.csv')
                    self.optimizer.params_rf.to_csv(f'Results/{self.dataset_name}/ModelOptimization/RandomForest_hyperparameters_metrics.csv')
                    self.optimizer.params_dt.to_csv(f'Results/{self.dataset_name}/ModelOptimization/DecisionTree_hyperparameters_metrics.csv')

        self.new_page()
        # add section with interpretabilty of the best models
        with self.doc.create(Section('Interpretabilty of the best models')):
            self.doc.append(NoEscape(r'Classify2TeX package defined the best model as the one that achievied the highest value of a metric, chosen by the user, or ROC AUC by default.'))
            self.doc.append(NoEscape(r'In this case, the optimization process was aimed at maximizing'))
            if self.optimizer.metric_to_eval == 'roc_auc':
                self.doc.append(NoEscape(r'\textbf{ ROC AUC.}'))
            elif self.optimizer.metric_to_eval == 'accuracy':
                self.doc.append(NoEscape(r'\textbf{ Accuracy.}'))
            elif self.optimizer.metric_to_eval == 'f1':
                self.doc.append(NoEscape(r'\textbf{ F1 Score.}'))
            self.doc.append(NoEscape(r'\\'))
            self.doc.append(NoEscape(r'Do not forget, that after preprocessing, columns names have changed, because of transformations of categorical features.'))
            with self.doc.create(Subsection('SHAP - what is under the hood?')):
                self.add_shap_description()

            self.new_page()
            with self.doc.create(Subsection('The best XGBoost model Explanation')):

                # add global feature importance using SHAP values 
                with self.doc.create(Subsubsection('XGBoost model - feature importance using SHAP values')):
                    self.add_shap_bar_plot_description()
                    self.explainer_best_xgb.save_global_feature_importance_shap(self.optimizer.X, self.dataset_name)
                    with self.doc.create(Figure(position='h!')) as fig:
                        fig.add_image(f'XAI/XGBoost/global_feature_importance_shap.png', width='350px')
                        fig.add_caption('SHAP values for the best XGBoost model')

                self.new_page()
                # add feature importance gained directly from the model
                with self.doc.create(Subsubsection('XGBoost model - feature importance gained directly from the model')):
                    self.add_feature_importance_plot_description()
                    self.explainer_best_xgb.save_feature_importance_plot(self.dataset_name)
                    with self.doc.create(Figure(position='h!')) as fig:
                        fig.add_image(f'XAI/XGBoost/feature_importance.png', width='350px')
                        fig.add_caption('Feature Importance for the best XGBoost model')

                self.new_page()
                # add violin plot of impact on prediction
                with self.doc.create(Subsubsection('XGBoost model - violin plot (SHAP) of impact on prediction')):
                    self.explainer_best_xgb.save_violin_summary_plot_shap(self.optimizer.X, self.dataset_name)
                    self.add_violin_plot_description()
                    with self.doc.create(Figure(position='h!')) as fig:
                        fig.add_image(f'XAI/XGBoost/violin_summary_plot_shap.png', width='400px')
                        fig.add_caption('Violin plot (SHAP) of impact on prediction for the best default XGBoost model')


        self.doc.generate_pdf(f'Results/{self.dataset_name}/report', clean_tex=False)


    

