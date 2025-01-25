# AutoPocket

_An automated machine learning package for financial analysis_


---

# 🌟 Introduction

This Python package helps users to quickly develop binary classification or regression models for financial data. It supports raw tabular data and provides comprehensive explanation of chosen algorithm.

## Target User Group
Autopocket is ideal for financial analysts, insurance actuaries, and data scientists working in regulated industries where model interpretability is crucial. It is also suitable for professionals with limited programming experience who need to develop predictive models quickly and accurately.

## Tool Specialization
The most important features of this package are prepocessing and postprocessing module. 

Our package is able to handle a wide variety of data disturbances and select the relevant columns and process them accordingly.

Autopocket also creates a report with many graphs explaining how the best model makes decisions based on different feature values ​​of the data. Every type of plot has proper description explaining it. 

> Note: Explanations are created only for supported models

The package is specialized in predictive analysis, specifically in binary classification and regression tasks. It is designed to handle financial data, providing detailed preprocessing, model training, and postprocessing steps. The tool focuses on generating explainable models, making it suitable for applications such as:
- Prediction of default of loans in banking.
- Estimation of insurance claims amounts.
- Credit risk assessment. 

## Comparison: Autopocket vs MLJAR and AutoGluon

In this section, we compare the functionality and benefits of Autopocket against two popular AutoML frameworks: **MLJAR** and **AutoGluon**. While each of these tools excels in specific areas, Autopocket introduces several unique advantages tailored for financial data analysis and model interpretability.

### 1. **Core Features**

| Feature                          | Autopocket                  | MLJAR (mode Explain)                            | AutoGluon (preset Medium Quality)                     |
|----------------------------------|--------------------------------|-----------------------------------|---------------------------------|
| **Model Types**                  | Supports explainable models like RandomForest, LogisticRegression, and DecisionTree. | Wide variety including ensembles, neural networks, etc. | Extensive set of models including XGBoost, CatBoost, and neural networks. |
| **Interpretability**             | Strong focus: SHAP, LIME, ICE, PDP. | SHAP support + learning curves. | Some SHAP support, but limited to computing feature importances, no plots generated. |
| **Optimization Strategy**        | Fine-tuning via RandomizedSearchCV with 5-fold cross-validation. | Automatic tuning with minimal user control. | Multi-level ensembling and stacking. |
| **Custom Preprocessing**         | Handles financial-specific issues (e.g., decimal separators, date formats). | General preprocessing pipeline. | Generic preprocessing for broader data types. |
| **Explanations and Visualizations** | Detailed explanations using LIME, SHAP, and dependency plots. | Limited visual explanations. | Focus on performance metrics; fewer visualizations. |
| **Output Format**                | Saves results in structured PDFs and JSON for further analysis, displaying some of the plots in Jupyter notebook cell. | Primarily charts saved in .png and .json files. | Emphasis on leaderboard-style reporting. |

### 2. **Performance and Usability**

| Feature                          | Autopocket                  | MLJAR (mode Explain)                            | AutoGluon (preset Medium Quality)                     |
|----------------------------------|--------------------------------|-----------------------------------|---------------------------------|
| **Ease of Use**                  | Very simple API: `doJob()` handles the entire pipeline. | User-friendly, but requires configuration for interpretability. | High automation, but API can be complex for customization. |
| **Performance Metrics**          | Predefined metric with financial relevance (e.g., gini for binary classification). | Predefined metrics for general-purpose tasks. | Comprehensive leaderboard with multiple metrics. |
| **Execution Time**               | Focuses on explainable models, trading speed for interpretability. | Fast due to Optuna optimization | Fast with multi-threaded optimization|

### 3. **Comparison by example on dataset credit.csv**

**Training restricted to tree-based and linear models due to banking regulations (no boostings)**

#### Autopocket:
- **Models Trained**: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, RidgeClassifier.
- **Best Model Score**: RandomForestClassifier with 0.8387 roc-auc.
- **Key Strengths**: Comprehensive interpretability via LIME and SHAP. Tailored for financial datasets, offering detailed preprocessing and explanations.
- **Use Case Fit**: Ideal for financial analysts requiring interpretability and actionable insights.

#### MLJAR (mode "Explain"):
- **Models Trained**: Baseline, DecisionTree, RandomForest, and Linear models.
- **Best Model Score**: RandomForest with roc-auc of 0.818378.
- **Key Strengths**: Speed and simplicity. Suitable for regression tasks with quick baseline comparisons.
- **Use Case Fit**: Great for rapid prototyping.

#### AutoGluon:
- **Models Trained**: RandomForestEntr, ExtraTreesEntr, ExtraTreesGini, RandomForestGini
- **Best Model Score**: RandomForestEntr with 0.851 (roc-auc).
- **Key Strengths**: High performance and ensembling. Designed for larger datasets with less concern for interpretability.
- **Use Case Fit**: Preferred for performance-oriented tasks without strict requirements for explanation.

### 4. **Why Autopocket is so unique**

1. **Interpretability First**:
   - Deep integration of SHAP, LIME, ICE, and PDP ensures models are explainable to non-technical stakeholders.
   - Autopocket emphasizes actionable insights and feature impact visualizations.

2. **Tailored for Financial Data**:
   - Handles domain-specific challenges like inconsistent date formats, decimal separators, and outlier detection.
   - Other frameworks offer generic preprocessing but lack domain-specific expertise.

3. **Simplified Workflow**:
   - A single `doJob()` function minimizes complexity while delivering high-quality results.
   - Users benefit from easy-to-understand outputs (PDFs, JSON) for decision-making.

4. **Focused Model Selection**:
   - Uses only interpretable and widely accepted models in the financial domain (e.g., RandomForest, LogisticRegression).
   - Avoids black-box models that are harder to justify in regulated environments.

### Conclusion

In a direct comparison on models compliant with the most stringent banking regulations (tree-based and linear models), AutoPocket achieved 2nd place in terms of the roc-auc metric, behind AutoGluon and ahead of MLJAR. While MLJAR and AutoGluon offer fast prototyping and high performance for general machine learning tasks, Autopocket carves out a niche by prioritizing **interpretability**, **financial data handling**, and **domain relevance**. It bridges the gap between performance and actionable insights, making it a valuable tool for financial analysts and decision-makers. 

---

# 📦 Installation
You can install this package using ``` pypi ```
```python
pip install autopocket
```
---

# 📖 User guide
The syntax is simple. Just import ```AutoPocketor``` and call ```doJob()``` with your data! For example:
```python
from autopocket import AutoPocketor

AutoPocketor().doJob(path = "path/to/your/data.csv", target = "target")
```
Our package will do rest for you


If you need some more personalization, you can also modify some of the parameters listed below:
```python
from autopocket import AutoPocketor

AutoPocketor().doJob(path = "path/to/your/data.csv",
                     target = "target", # target column name
                     num_strategy='mean', # strategy for imputing missing data in numerical columns
                     cat_strategy='most_frequent', # strategy for imputing missing data in categorical columns
                     fill_value=None, # optional fixed value to fill missing data if cat_strategy is "constant"
                     generate_shap=True, # generate SHAP values and plots
                     generate_lime_pdp_ice=True, # generate LIME, PDP and ICE plots
                     features_for_displaying_plots=None, # list of features for displaying ICE and PDP plots
                     subset_fraction_for_ICE_plot=None, # fraction of data to be used for ICE plot
                     additional_estimators=None # additional estimators to be used for SHAP values
                     )

```

---

# 🤔 How it works?
We have divided the whole process into three main parts:
* Preprocessing
* Algorithms
* Postprocessing

You can find explanation of each part below:


---

# 🔄 Preprocessing

Preprocessing handles and adjusts data with imperfections such as inconsistent date formats, repetitive patterns, unnecessary spaces, missing data, alternating use of commas and periods as decimal separators, redundant columns, inconsistent binary variable formats, and variations in text case. The preprocessing module is divided into three main components: ```Task Analysis```, ```Data Cleaning``` and ```Feature Processing``` which are described below.  

**`Preprocessor`** is a class that combines **Task Analysis**, **Data Cleaning** and **Feature Processing** into a single workflow.


## 🔍 Task Analysis

The `task_analysing` module includes **`ColumnTypeAnalyzer`** which is responsible for determining the type of machine learning task: **binary classification** or **linear regression**. Here, too, the target variable is transformed. In the case of **binary classification**, it is transformed to values 0 and 1, while in the case of **linear regression**, it is transformed to have a normal distribution.

## 🧹 Data Cleaning

The `data_cleaning` module is responsible for handling missing values, correcting data formats, and removing unnecessary or redundant columns.

Tools available in this module:

| Tool                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `BinaryColumnHandler`   | Ensures consistency in binary column encoding.                              |
| `DataImputer`           | Fills missing values using predefined strategies (mean, median, mode, etc.).|
| `DateHandler`           | Processes date columns, handles different data formats.                     |
| `NumberFormatFixer`     | Fixes inconsistent number formats (e.g., commas vs. periods as separators). |
| `PatternRemover`        | Removes unwanted, repeated patterns.                                        |
| `RedundantColumnRemover`| Identifies and removes redundant columns (e.g., index or same value).       |
| `StringStripper`        | Cleans string columns by removing leading/trailing whitespace.              |
| `StringToLowerConverter`| Converts string data to lowercase for consistency.                          |

All these tools are managed by the `DataCleaner` class, which applies the cleaning steps sequentially.



## ⚙️ Feature Processing

The `feature_processing` module focuses on transforming the cleaned data into a format suitable for machine learning algorithms. It includes the following tools:

| **Class**           | **Description**                                                                                          |
|---------------------|----------------------------------------------------------------------------------------------------------|
| **`FeatureEncoder`** | Encodes categorical features using one-hot encoding and label encoding for binary features.              |
| **`FeatureSelector`**| Selects relevant features by removing highly correlated ones.                                            |
| **`OutlierHandler`** | Detects and handles outliers using methods like IQR or Isolation Forest.                                  |

The `FeatureProcessor` class manages these steps in data processing.


## ✨ Example Usage of `Preprocessor`

Below is an example of how to use the `Preprocessor` class to complete preprocessing:

```python
from autopocket.preprocessing.Preprocessor import Preprocessor

# Run preprocessing
X, y, ml_type = Preprocessor().preprocess(path="path/to/your/data.csv", target="your_target_column")
```


---

# 🤖 Algorithms

When preprocessinng is done, the ```Modeller``` chooses the set of used models according to recognized task.
We only use those models that are meant to be explainable and are refered to be commonly used in finance. 

We implement estimators form ```scikit-learn``` package

In both cases, we finetune models using ```sklearn.model_selection.RandomizedSearchCV``` (```LassoLarsIC``` and ```LinearRegression``` are tuned using ```sklern.model_selection.GridSearchCV``` because of small hyperparameters space) on 5 folds. We pick number of iterations depending on the model. 

The best model is the one with the highest score in _proper_ metric.

Before starting fitting process, there is information about baseline model score. Baseline (dummy) strategy is chosen depending on the task type. Baseline predicts constant value for every obseravation in the whole dataset and then the score is measured using true target value.

## 🧩 Binary classification

In this task, we use _roc-auc_ score to measure estimators' performance. 
Baseline predicts the most frequent value present in target value (```y```) for every observation.

We use the following estimators:
* LogisticRegression
* RandomForestClassifier
* RidgeClassifier
* DecisionTreeClassifier


## 📈 Regression

In this task, we use _negative root mean squared error_ score to measure estimators' performance. 
Baseline predicts mean of target value (```y```) for every observation.

We use the following estimators:
* LinearRegression (implementation od Least Squares)
* Lasso
* Ridge
* DecisionTreeRegression
* RandomForestRegression
* ElasticNet
* LassoLarsIC

Results from this part are saved in ```.json``` files in ```algorithms_results_%Y%m%d_%H%M%S``` directory (where %Y, %m, %d, %H, %M, %S stands for year, month, day ,hour, month, second respectively- the time when you call ```doJob()```). 

You can load results using ```autopocket.algorithms.utils.ResultsReader.results``` into python ```dict```.

## Training aditional models
One may want to fit additional estimators. It is possible by settind ```aditional_estimators``` parameter, which should be a ```list``` of ```EstimatorWrapper```. You can easly create ```EstimatorWrapper``` using ```autopocket.algorithms.base.create_wrapper(estimator, param_distributions, name, n_iter)``` function.


---

# 💡 Postprocessing

The **Postprocessing** module focuses on interpreting and visualizing the results of the best-performing machine learning model. This step provides insights into model behavior, feature importance, and the decisions it makes.



## 📊 LIME Explanations

LIME (Local Interpretable Model-Agnostic Explanations) provides localized explanations for predictions by approximating the model with interpretable surrogates.

| **Function**                         | **Description**                                                                                                    |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `explain_top_observations_with_lime` | Generates LIME explanations for the most influential observations.                                                 |
| `lime_summary_plot`                  | Aggregates feature importance from multiple LIME explanations into a global summary plot.                          |
| `top_features_by_lime_importance`    | Selects top uncorrelated features based on LIME feature importance for detailed visualization and model insights.   |


## ✨ SHAP Explanations

SHAP (SHapley Additive exPlanations) is used for interpreting global and local model behavior, leveraging Shapley values.

| **Function**               | **Description**                                                                                       |
|----------------------------|-------------------------------------------------------------------------------------------------------|
| `explain_with_shap`        | Performs a full SHAP analysis, including summary plots, dependence plots, decision plots, and more.    |
| `shap_summary_plot`        | Visualizes global feature importance using SHAP values. The plot highlights which features contribute most to the model's output, based on the magnitude and distribution of SHAP values across all samples.                                               |
| `shap_dependence`          | Displays interactions between features using dependence plots. These plots show the relationship between a feature's value and its SHAP value, revealing potential interactions.                                    |
| `decisions_binary`         | Creates SHAP decision plots for binary classification tasks, focusing on both best and worst decisions. Best decisions correspond to the highest predicted probabilities, while worst decisions have the largest residuals.|
| `decisions_regression`     | Creates SHAP decision plots for regression tasks. The plots identify instances with the highest and lowest residuals to illustrate model performance.                                                     |
| `forceplot_binary`         | Generates SHAP force plots for individual predictions in binary classification. Force plots are created for the observation with the highest probability for class 1 and the observation with the highest probability for class 0, showing local contributions to the prediction.                       |


## 📈 Partial Dependence and ICE Plots

Partial Dependence Plots (PDPs) and Individual Conditional Expectation (ICE) Plots provide visualizations of the relationship between feature values and model predictions. These visualizations help interpret the global and local impacts of features on model predictions.

| **Function**              | **Description**                                                                                               |
|---------------------------|---------------------------------------------------------------------------------------------------------------|
| `generate_pdp`            | Creates Partial Dependence Plots (PDPs) for the selected features, visualizing their average influence on model predictions. By default, the most important, uncorrelated, non-binary features are shown, but user can specify which features should be visualized on plots.
| `generate_ice`            | Produces Individual Conditional Expectation (ICE) plots for visualizing the effect of feature changes on individual predictions. By default, the most important, uncorrelated, non-binary features are shown, but user can specify which features should be visualized on plots. Supports limiting the number of data samples for better visibility. |

These functions include robust error handling to ensure that invalid or unsupported feature names are excluded and provide meaningful feedback to guide users during plot generation.

## 🏆 Model Leaderboard

This component compares model performance and provides a ranked summary of all evaluated models.

| **Function**       | **Description**                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| `createLeaderBoard`| Reads the saved model results and generates a leaderboard summarizing performance. |


This module ensures transparency and interpretability, making the models ready for critical financial decisions.

# 📦 Dependencies
```plaintext
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5
matplotlib>=3.10
joblib>=1.4
lime>=0.2
shap>=0.46
openml>=0.15
requests>=2.32
xmltodict>=0.14
```

