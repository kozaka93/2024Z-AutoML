from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot as plt



def get_param_score_list(X_train, y_train, X_test, y_test, estimator, param_distributions, n):
    """

    :param X_train: Dataset for model training
    :param y_train: Target variable
    :param X_test:  Dataset for model testing
    :param y_test: Target variable
    :param estimator: model used for regression with its own pipeline
    :param param_distributions: net for 3 parameters for randomized search
    :param n: total number of iterations
    :return:
    param_list, score_list: best params for every iteration with mse score
    """
    score_list = []
    param_list = []

    for i in range(2,10,2):
        random_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions, n_iter=i, cv=5, n_jobs=-1,
                                           verbose=1, scoring='r2', random_state=42)
        random_search.fit(X_train, y_train)
        param_list.append(random_search.best_params_)
        score_list.append(random_search.score(X_test, y_test))

    for i in range(10, n, 10):
        random_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions, n_iter=i, cv=5, n_jobs=-1,
                                           verbose=1, scoring='r2', random_state=42)
        random_search.fit(X_train, y_train)
        param_list.append(random_search.best_params_)
        score_list.append(random_search.score(X_test, y_test))

    print(param_list)
    print(score_list)

    return param_list, score_list

def r2_iteration_plot(iteration_vector, values, dataset_name, model_name):

    save_path = '../figures'
    plt.plot(iteration_vector, values, marker='o', linestyle='-', color='b', label='R^2')

    plt.title('Wartość R^2 w zależności od liczby iteracji')
    plt.xlabel('Iteracje')
    plt.ylabel('R^2')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_path}/{dataset_name}_{model_name}.png", format="png")
    plt.show()

def r2_all_models(iteration_vector, rf_values,xgb_values, enf_values, dataset_name):

    save_path = '../figures'
    plt.plot(iteration_vector, rf_values, label='rf', color='blue')
    plt.plot(iteration_vector, xgb_values, label='xgb', color='green')
    plt.plot(iteration_vector, enf_values, label='enet', color='red')

    # Dodanie tytułu, legendy i etykiet
    plt.title('Wartość R^2 w zależności od liczby iteracji')
    plt.xlabel('Iteracje')
    plt.ylabel('R^2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{dataset_name}_all_models.png", format="png")
    plt.show()