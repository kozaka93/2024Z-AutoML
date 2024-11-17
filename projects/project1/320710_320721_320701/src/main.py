from data_loader import (
    load_wine_data,
    load_drug_data,
    load_iris_data,
    load_titanic_data,
)
from preprocessing import (
    create_wine_preprocessor,
    create_drug_preprocessor,
    create_iris_preprocessor,
    create_titanic_preprocessor,
)
from model_tuning import (
    create_rf_pipeline,
    create_lr_pipeline,
    create_xg_pipeline,
    perform_grid_search,
    perform_random_search,
    perform_bayesian_search,
    get_rf_search_spaces,
    get_lr_search_spaces,
    get_xg_search_spaces,
)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

def statistical_tests(results):
    p_values_data = []
    for dataset_name, dataset in results.items():
        for model_name, model in dataset.items():
            methods = ["Grid Search", "Random Search", "Bayesian Search"]
            scores = {method: model[method]["Test Scores"] for method in methods}
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method_i = methods[i]
                    method_j = methods[j]
                    scores_i = scores[method_i]
                    scores_j = scores[method_j]
                    try:
                        _, p_value = wilcoxon(scores_i, scores_j)
                        conclusion = (
                            "Istnieją istotne różnice między próbkami"
                            if p_value < 0.05
                            else "Brak istotnych różnic między próbkami"
                        )
                    except ValueError:
                        p_value = np.nan
                        conclusion = "Test not applicable"
                    p_values_data.append(
                        {
                            "Algorytm": model_name,
                            "Zbiór danych": dataset_name,
                            "Technika 1": method_i,
                            "Technika 2": method_j,
                            "p-value": p_value,
                            "Wniosek": conclusion,
                        }
                    )
    p_values_df = pd.DataFrame(p_values_data)
    print("\nWyniki testów statystycznych:")
    print(p_values_df)
    return p_values_df

def run_analysis(models, datasets, n_runs=3):
    all_results = {}
    convergence_info = {}
    for dataset_name, X, y, preprocessor in datasets:
        results = {}
        for model_name, create_pipeline, param_grid, search_spaces in models:
            print(f"\nAnaliza {model_name} na zbiorze {dataset_name}:")
            pipeline = create_pipeline(preprocessor)
            scores = {
                "Grid Search": [],
                "Random Search": [],
                "Bayesian Search": [],
            }
            convergence_scores = {
                "Grid Search": [],
                "Random Search": [],
                "Bayesian Search": [],
            }
            for run in range(n_runs):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42 + run
                )
                grid_search = perform_grid_search(pipeline, param_grid, X_train, y_train)
                random_search = perform_random_search(pipeline, param_grid, X_train, y_train)
                bayesian_search = perform_bayesian_search(
                    pipeline, search_spaces, X_train, y_train
                )
                grid_test_score = grid_search.best_estimator_.score(X_test, y_test)
                random_test_score = random_search.best_estimator_.score(X_test, y_test)
                bayesian_test_score = bayesian_search.best_estimator_.score(X_test, y_test)
                scores["Grid Search"].append(grid_test_score)
                scores["Random Search"].append(random_test_score)
                scores["Bayesian Search"].append(bayesian_test_score)
                convergence_scores["Grid Search"].append(grid_search.cv_results_["mean_test_score"])
                convergence_scores["Random Search"].append(random_search.cv_results_["mean_test_score"])
                convergence_scores["Bayesian Search"].append(bayesian_search.cv_results_["mean_test_score"])
            results[model_name] = {
                "Grid Search": {
                    "Test Scores": scores["Grid Search"],
                    "Convergence Scores": convergence_scores["Grid Search"],
                },
                "Random Search": {
                    "Test Scores": scores["Random Search"],
                    "Convergence Scores": convergence_scores["Random Search"],
                },
                "Bayesian Search": {
                    "Test Scores": scores["Bayesian Search"],
                    "Convergence Scores": convergence_scores["Bayesian Search"],
                },
            }
            if model_name not in convergence_info:
                convergence_info[model_name] = {}
            convergence_info[model_name][dataset_name] = {
                "Grid Search": convergence_scores["Grid Search"],
                "Random Search": convergence_scores["Random Search"],
                "Bayesian Search": convergence_scores["Bayesian Search"],
            }
            print(f"Średnie wyniki testowe dla {model_name} na {dataset_name}:")
            print(f"  Grid Search Test Score: {np.mean(scores['Grid Search']):.4f}")
            print(f"  Random Search Test Score: {np.mean(scores['Random Search']):.4f}")
            print(f"  Bayesian Search Test Score: {np.mean(scores['Bayesian Search']):.4f}")
        all_results[dataset_name] = results
    return all_results, convergence_info

def plot_convergence(convergence_info):
    for model_name, datasets_data in convergence_info.items():
        methods = ["Grid Search", "Random Search", "Bayesian Search"]
        for method in methods:
            plt.figure(figsize=(10, 6))
            for dataset_name, method_data in datasets_data.items():
                convergence_scores_runs = method_data[method]
                max_length = max(len(scores) for scores in convergence_scores_runs)
                padded_scores = [np.pad(scores, (0, max_length - len(scores)), 'edge') for scores in convergence_scores_runs]
                mean_convergence_scores = np.mean(padded_scores, axis=0)
                cum_best_scores = np.maximum.accumulate(mean_convergence_scores)
                plt.plot(
                    range(1, len(cum_best_scores) + 1),
                    cum_best_scores,
                    marker="o",
                    label=dataset_name,
                )
            plt.xlabel("Iteracja")
            plt.ylabel("Skumulowany najlepszy średni wynik CV")
            plt.title(f"Konwergencja dla {model_name} z użyciem {method}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{model_name}_{method}_convergence.png")
            plt.close()

def plot_boxplots(results):
    for dataset_name, dataset in results.items():
        for model_name, model in dataset.items():
            methods = ["Grid Search", "Random Search", "Bayesian Search"]
            data = []
            for method in methods:
                scores = model[method]["Test Scores"]
                data.extend([(method, score) for score in scores])
            df = pd.DataFrame(data, columns=["Technika", "Wynik testowy"])
            plt.figure(figsize=(8, 6))
            sns.boxplot(x="Technika", y="Wynik testowy", data=df)
            plt.title(f"Boxplot wyników dla {model_name} na {dataset_name}")
            plt.ylabel("Wynik testowy")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{model_name}_{dataset_name}_boxplot.png")
            plt.close()

def main():
    wine_X, wine_y = load_wine_data()
    wine_preprocessor = create_wine_preprocessor()
    drug_X, drug_y = load_drug_data()
    drug_preprocessor = create_drug_preprocessor()
    iris_X, iris_y = load_iris_data()
    iris_preprocessor = create_iris_preprocessor()
    titanic_X, titanic_y = load_titanic_data()
    titanic_preprocessor = create_titanic_preprocessor()
    datasets = [
        ("Wine", wine_X, wine_y, wine_preprocessor),
        ("Drug", drug_X, drug_y, drug_preprocessor),
        ("Iris", iris_X, iris_y, iris_preprocessor),
        ("Titanic", titanic_X, titanic_y, titanic_preprocessor),
    ]
    models = [
        ("Random Forest", create_rf_pipeline, *get_rf_search_spaces()),
        ("Logistic Regression", create_lr_pipeline, *get_lr_search_spaces()),
        ("XGBoost", create_xg_pipeline, *get_xg_search_spaces()),
    ]
    all_results, convergence_info = run_analysis(models, datasets, n_runs=3)
    p_values_df = statistical_tests(all_results)
    p_values_df.to_csv("statistical_test_results.csv", index=False)
    plot_convergence(convergence_info)
    plot_boxplots(all_results)

if __name__ == "__main__":
    main()
