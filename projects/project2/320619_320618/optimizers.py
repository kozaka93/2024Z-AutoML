from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
import numpy as np
from models_config import get_models, get_param_grids
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import pandas as pd

class Optimizer:
    def __init__(self, metric, optimizer, cv, n_iter, seed, n_jobs):
        self.metric = metric
        self.optimizer = optimizer
        self.cv = cv
        self.n_iter = n_iter
        self.seed = seed
        self.n_jobs = n_jobs
        self.scores = pd.DataFrame(columns=["Model", "Metric", "Best Score", "Best Params"])
        self.best_models = {}
        self.iteration_scores = {}

        self.optimizers_map = {
            'random': RandomizedSearchOptimizer,
            'bayes': BayesSearchOptimizer
        }

        self.supported_metrics = {
            'accuracy': accuracy_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            'precision': precision_score,
            'recall': recall_score
        }

        if self.optimizer not in self.optimizers_map:
            raise ValueError("Unsupported optimizer. Choose 'random' or 'bayes'.")

        if self.metric not in self.supported_metrics:
            raise ValueError(f"Unsupported metric. Choose from {list(self.supported_metrics.keys())}.")

    def optimize(self, X, y):
        models = get_models(seed=self.seed)
        param_grids = get_param_grids()
        best_score = -np.inf
        best_model = None
        best_params = None

        optimizer_class = self.optimizers_map[self.optimizer]

        for model_name, model in models.items():
            print(f"Optimizing model: {model_name}...")
            opt = optimizer_class(
                param_space=param_grids[model_name],
                scoring=self.metric,
                cv=self.cv,
                n_iter=self.n_iter,
                random_state=self.seed,
                n_jobs=self.n_jobs
            )
            best_estimator, current_params, current_score = opt.optimize(model, X, y)

            print(f"Finished optimization for model: {model_name}")
            print(f"Best {self.metric}: {current_score:.4f}")

            self.scores = pd.concat(
                [self.scores, pd.DataFrame([{
                    "Model": model_name,
                    "Metric": self.metric,
                    "Best Score": round(current_score, 4),
                    "Best Params": current_params
                }])],
                ignore_index=True
            )

            self.best_models[model_name] = {
                "best_model": best_estimator,
                "best_params": current_params,
                "best_score": round(current_score, 4)
            }

            if current_score > best_score:
                best_score = current_score
                best_model = best_estimator
                best_params = current_params

        return best_model, best_params, best_score


class RandomizedSearchOptimizer:
    def __init__(self, param_space, scoring, cv, n_iter, random_state, n_jobs):
        self.param_distributions = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def optimize(self, model, X, y):
        optimizer = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.param_distributions,
            scoring=self.scoring,
            cv=self.cv,
            n_iter=self.n_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        optimizer.fit(X, y)
        return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_


class BayesSearchOptimizer:
    def __init__(self, param_space, scoring, cv, n_iter, random_state, n_jobs):
        self.search_spaces = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def optimize(self, model, X, y):
        optimizer = BayesSearchCV(
            estimator=model,
            search_spaces=self.search_spaces,
            scoring=self.scoring,
            cv=self.cv,
            n_iter=self.n_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        optimizer.fit(X, y)
        return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_
