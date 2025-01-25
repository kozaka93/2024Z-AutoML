import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class ModelOptimizer:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.features = [col for col in df.columns if col != target]
        x_train, x_test, y_train, y_test = train_test_split(self.df[self.features], self.df[self.target], stratify=self.df[self.target], test_size=0.3,
                                                  random_state=10)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def get_models(self):
        models = dict()
        models['lr'] = LogisticRegression()
        models['cart'] = DecisionTreeClassifier()
        models['rf'] =  RandomForestClassifier(random_state=1)
        models['knn'] = KNeighborsClassifier()
        models['xgboost'] = XGBClassifier(random_state=1)
        models['bayes'] = GaussianNB()

        return models

    def get_param_grid(self):

        param_grids = {
            'lr': {
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs', 'saga']
            },
            'cart': {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.5, 0.7, 1.0],
                'colsample_bytree': [0.5, 0.7, 1.0]
            },
            'bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        }

        return param_grids

    def evaluate_model(self, X, y, model, metric = 'accuracy'):

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1, error_score='raise')

        return scores

    def get_models_candidates(self):

        models = self.get_models()
        results, names = list(), list()
        for name, model in models.items():
            model.fit(self.x_train, self.y_train)
            scores = self.evaluate_model(self.x_test, self.y_test, model)
            results.append(scores)
            names.append(name)

        plt.figure(figsize=(12, 8))
        plt.boxplot(results, labels=names, showmeans=True)
        plt.title('Model comparison')
        plt.savefig("./figures/boxplot.png", bbox_inches="tight")
        plt.show()



    def get_param_score_list(self, estimator, param_distributions, n):

        random_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions, n_iter=n,
                                           cv=5, n_jobs=-1,
                                           verbose=1, scoring='accuracy', random_state=42)
        random_search.fit(self.x_train, self.y_train)

        return random_search.best_params_, random_search.score(self.x_test, self.y_test)

    def optimize_model(self, model):

        assert model in self.get_models().keys(), f'model name should be in {self.get_models().keys()}'

        estimator = self.get_models()[model]
        param_grid = self.get_param_grid()[model]
        param_list, score_list = self.get_param_score_list(estimator, param_grid, 100)

        return estimator.set_params(**param_list)