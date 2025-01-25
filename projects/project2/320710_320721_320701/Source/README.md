# AutoMLClassifier 🚀
**Automation of machine learning with visualizations and report generation**

AutoMLClassifier is a library that simplifies the process of building classification models in machine learning. It integrates data preprocessing, model selection, hyperparameter tuning, and evaluation report generation into a single, intuitive tool.

---

## 📦 Features
- **Data Preprocessing**
  - Handles missing data.
  - Scales numerical variables.
  - Encodes categorical variables.

- **Automatic Model Selection**
  - Supports models such as:
    - Random Forest
    - Decision Tree
    - Support Vector Machine (SVM)
    - XGBoost
  - Enables hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.

- **Model Evaluation**
  - Automatically generates detailed evaluation reports in HTML.
  - Includes visualizations like confusion matrices.

- **Clear Result Visualization**
  - Tools for tracking model convergence and comparing performance.

---

## 📥 Installation
### Install locally:
```bash
pip install Source/dist/automlclassifier-0.1.0-py3-none-any.whl
```
### Install from PyPI:
```bash
pip install automlclassifier
```

---

## 🔧 Requirements
- Python >= 3.7
- Installed libraries:
  - `scikit-learn>=1.0`
  - `pandas>=1.3`
  - `numpy>=1.21`
  - `matplotlib>=3.4`
  - `seaborn>=0.11`
  - `xgboost>=1.5`

---
