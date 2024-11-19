# Models to select from

## KNeighborsClassifier +
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
sensitive for hyperparams, quite simple, not so good for large datasets (quite slow)

**Key Hyperparameters**:
  - Number of neighbors \( n\_neighbors \)
  - Distance metric \( metric \)
  - Weighting of neighbors \( weights \)


## DecisionTreeClassifier +
tree \
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

**Key Hyperparameters**:
  - Max depth \( max\_depth \)
  - much more


## GaussianProcessClassifier
Gaussian process classification (GPC) based on Laplace approximation. \
https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html

**Key Hyperparameters**:
  - Max iter predict \( max\_iter\_predict \)
  - much more


## PassiveAggressiveClassifier
Passive Aggressive Classifier. \
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html

**Key Hyperparameters**:
  - C \( C \)
  - much more


## XGBoost +
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html \
a boosting algorithm, popular for classification tasks, has a lot of hyperparams

**Key Hyperparameters**:
  - Learning rate \( learning\_rate \)
  - Number of trees \( n\_estimators \)
  - Maximum depth \( max\_depth \)
  - Regularization \( lambda \), \( alpha \)


## Naive Bayes
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html \
naive and dumb and lacking hyperparams model but Filip still wants it  
assumes feature independence, so it is good only for data with independent features, there are multiple versions of NB with different hiperparams

**Key Hyperparameters**:
- Laplace smoothing parameter (alpha)


## Random Forest 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html \
uses multiple decision tress, has many hiperparams to tune

**Key Hyperparameters**:
  - Number of trees \( n\_estimators \)
  - Maximum depth \( max\_depth \)
  - Maximum features \( max\_features \)
  - Minimum samples per leaf \( min\_samples\_leaf \)


## SVM
a good model in general, good for not easily separable data, good for small and medium-sized datasets

**Key Hyperparameters**: 
  - Regularization parameter \( C \)
  - Kernel \( \gamma \) (for RBF kernel)
  - Kernel type (linear, RBF, polynomial)


## Neural Network (MLPClassifier)
high tunability according to the Internet

**Key Hyperparameters**:
  - Number of layers
  - Number of neurons per layer
  - Learning rate \( learning\_rate \)
  - Momentum \( momentum \)


## LightGBM
gradient boosting method, good for large datasets, has many hyperparams

**Key Hyperparameters**:
  - Learning rate \( learning\_rate \)
  - Number of leaves \( num\_leaves \)
  - Minimum data in leaf \( min\_data\_in\_leaf \)
  - Number of iterations \( n\_estimators \)