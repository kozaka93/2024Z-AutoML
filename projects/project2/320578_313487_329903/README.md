# Classify2TeX

Classify2TeX is an all-in-one automated machine learning tool for binary classification, hyperparameter optimization, and professional reporting. Designed to streamline machine learning workflows, it handles every step from preprocessing data, through hyperparameter optimization to generating comprehensive reports in LaTeX and PDF formats. The tool emphasizes interpretability, robust analysis, and ease of use, making it ideal for academic, research, and business applications.

# Key Features

## 1. Automated Data Preprocessing

Classify2TeX simplifies data preparation with a fully automated preprocessing pipeline:

•	Extracts Date Components: Splits datetime features into separate day, month, and year components.

•	Removes Redundant Features: Eliminates unsupported or uninformative columns and those with excessive missing values.

•	Handles Missing Data: Imputes missing values based on feature type (mode for categorical, median for numerical).

•	Manages Outliers: Detects and mitigates outliers in numerical features using the Z-score method.

•	Encodes Categorical Variables: Applies one-hot or label encoding for compatibility with machine learning models.

•	Transforms Boolean Features: Converts boolean features into integer representations.

•	Encodes the Target Variable: Prepares the target column for machine learning.

•	Removes Highly Correlated Features: Identifies and drops features with high correlation to reduce redundancy.

•	Balances Classes: Addresses class imbalances with appropriate resampling techniques (oversampling, undersampling, or SMOTE).


## 2. Comprehensive Model Optimization

Leverage hyperparameter tuning and evaluation techniques:

•	Random Search Optimization: Tunes key hyperparameters for Random Forest, Decision Tree, and XGBoost models.

•	Robust Evaluation: Uses cross-validation and repeated cross-validation to ensure stability and reliability.

•	Metric-Driven Insights: Focuses on key metrics like ROC AUC, F1 score, and accuracy for model comparison.

## 3. Insightful Visualization and Explainability

•	Performance Graphs: Visualizes model performance across metrics for intuitive comparisons.

•	Decision Tree Explanation: Builds and visualizes the best Decision Tree model, highlighting feature importance and splits for interpretability.

## 4. Professional LaTeX and PDF Reporting

Generate high-quality reports:

•	Dataset Analysis: Includes summaries and preprocessing steps.

•	Model Insights: Details strengths, weaknesses, and optimal hyperparameters for each model.

•	Visual Results: Incorporates graphs and tables with results for clear communication of findings.

•	Seamless PDF Conversion: Produces ready-to-use reports in both PDF and Latex format.

# Examples

See [this file](https://github.com/kateqwerty001/Classify2TeX/blob/main/TUTORIAL.ipynb) - User Guide part explains on example how to use Classify2TeX.

See [Results](https://github.com/kateqwerty001/Classify2TeX/tree/main/Results) folder for example of generated reports and folders on 4 different datasets.

# Review of existing solutions

See [this file](https://github.com/kateqwerty001/Classify2TeX/blob/main/review_of_existing_solutions.ipynb)

# What makes Classify2Tex stand out?

Comprehensive Reporting: Generates detailed reports in both LaTeX and PDF formats, perfect for scientific papers, professional documentation, or quick analysis needs.

User-Friendly for All: Designed for both experienced machine learning practitioners and newcomers, ensuring ease of use without sacrificing depth.

Scientific-Ready Outputs: LaTeX reports integrate seamlessly into academic workflows, providing polished, citation-ready content.

Extensive Insights: Delivers solid information on preprocessing steps, model performance, and hyperparameter optimization, aiding in well-informed decision-making.

Streamlined Model Selection: Simplifies choosing the best model for binary classification by presenting key metrics like ROC AUC, F1 score, and accuracy in a clear, accessible manner.

Time-Saving Automation: Eliminates the need for manual preprocessing, optimization, and reporting, accelerating the workflow for all users.

# Authors:

Katsiaryna Bokhan

Dorota Rzewnicka

Monika Jarosińska
