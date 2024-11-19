# Hyperparameter Tuning and Tunability Calculations

## Project Overview

This project focuses on tuning hyperparameters for three machine learning algorithms: `LogisticRegression`, `ExtraTrees`, and `XGBoost`, using two optimization techniques—`Random Search (RS)` and `Bayesian Search (TPE)`—via the `Optuna` library. The tuning process is conducted on four different datasets, and the results (history of tuning, best parameters, performance metrics) are stored for further analysis. The tunability of the algorithms is computed based on the definitions provided in the paper *Tunability: Importance of Hyperparameters of Machine Learning Algorithms*.

The entire process is divided into two major phases:
1. **Hyperparameter Optimization** using Optuna (in `Hyperparametrs_in_optuna.ipynb`).
2. **Analysis of Tuning Results** (in `Analysis.ipynb`).


## Dependencies
Ensure that the required Python packages are installed by using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Notebooks 
1. Hyperparameter Optimization (Hyperparametrs_in_optuna.ipynb)
This notebook is responsible for running hyperparameter optimization using two methods:

- Random Search (RS)
- Tree-structured Parzen Estimator (TPE)
  
For each of the three models (LogisticRegression, ExtraTrees, XGBoost) and four datasets, the hyperparameters are tuned using the Optuna library. The optimization process logs:

Tuning history: Stored as CSV files in results/.
Best hyperparameters: Stored as JSON files in results/best_params/.

Running the notebook:
Ensure that your environment is set up with the required dependencies.
Run the notebook cell by cell to perform hyperparameter tuning on each model and dataset.
The results will automatically be saved in the results/ folder.


2. Analysis of Tuning Results (Analysis.ipynb)
This notebook reads the tuning results generated in the previous step and performs the following analyses:

- Tunability calculation: Based on the differences in performance (AUC) between the default and the tuned hyperparameters.
- Best AUC analysis: Compares the best AUC obtained from tuning versus package defaults.
- Time analysis: Evaluates the time taken for optimization using different techniques.
- Visualizations: Generates plots that summarize the tuning process and results.

All the results from the analysis, including the computed tunability values, visualizations, and tables, are saved in the results/ folder.

Running the notebook:
Ensure that the tuning results (CSV and JSON files) from Hyperparametrs_in_optuna.ipynb are present in the results/ folder.
Run the notebook to load and analyze the tuning history and best hyperparameters.
Generated plots and tables will be saved in the respective subfolders of results/.
