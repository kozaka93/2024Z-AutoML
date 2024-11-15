import pandas as pd
import os
import pickle
import json

results_dir = os.path.join(os.getcwd(), 'results')
results_studies_dir = os.path.join(results_dir, 'studies')
results_bestparams_dir = os.path.join(results_dir, 'best_params')

def save_results(results: pd.DataFrame, output_file_path: str):
    results.to_csv(output_file_path)

def save_study_to_pickle_joint(study, model_short_name):
    """
    Save an Optuna study to a pickle file for joint optimization.
    
    Args:
        study: Optuna study object to be saved.
        model_short_name: One of ('lr', 'et', 'xgb') representing the model.
    """
    filename = f"{model_short_name}_joint_study.pkl"
    with open(os.path.join(results_studies_dir, filename), 'wb') as f:
        pickle.dump(study, f)


def save_study_to_pickle_marginal(study, model_name, sampler_name, dataset_id):
    """
    Save an Optuna study to a pickle file for marginal optimization.
    
    Args:
        study: Optuna study object to be saved.
        model_name: Name of the model. One of ('lr', 'et', 'xgb').
        sampler_name: Name of the sampler. One of ('RS', 'BS')
        dataset_id: ID of the dataset.
    """
    filename = f"{model_name}_{sampler_name}_ID{dataset_id}_marginal_study.pkl"
    with open(os.path.join(results_studies_dir, filename), 'wb') as f:
        pickle.dump(study, f)


def save_best_params_to_json_joint(best_params, model_short_name):
    """
    Save best parameters to a json file for joint optimization.
    
    Args:
        best_params: Dictionary of best parameters to be saved.
        model_short_name: One of ('lr', 'et', 'xgb') representing the model.
    """
    filename = f"{model_short_name}_best_params_joint.json"
    
    # Save best params to a json file
    with open(os.path.join(results_bestparams_dir, filename), 'w') as f:
        json.dump(best_params, f)



def save_best_params_to_json_marginal(best_params, model_name, sampler_name, dataset_id):
    """
    Save best parameters to a json file for marginal optimization.
    
    Args:
        best_params: Dictionary of best parameters to be saved.
        model_name: Name of the model. One of ('lr', 'et', 'xgb').
        sampler_name: Name of the sampler. One of ('RS', 'BS')
        dataset_id: ID of the dataset.
    """
    filename = f"{model_name}_{sampler_name}_ID{dataset_id}_best_params_marginal.json"
    with open(os.path.join(results_bestparams_dir, filename), 'w') as f:
        json.dump(best_params, f)

def read_study_from_pickle_joint(model_short_name):
    """
    Read an Optuna study from a pickle file for joint optimization.
    
    Args:
        model_short_name: One of ('lr', 'et', 'xgb') representing the model.
    
    Returns:
        The Optuna study object.
    """
    filename = f"{model_short_name}_joint_study.pkl"
    with open(os.path.join(results_studies_dir, filename), 'rb') as f:
        return pickle.load(f)


def read_study_from_pickle_marginal(model_name, sampler_name, dataset_id):
    """
    Read an Optuna study from a pickle file for marginal optimization.
    
    Args:
        model_name: Name of the model. One of ('lr', 'et', 'xgb').
        sampler_name: Name of the sampler. One of ('RS', 'BS')
        dataset_id: ID of the dataset.
    
    Returns:
        The Optuna study object.
    """
    filename = f"{model_name}_{sampler_name}_ID{dataset_id}_marginal_study.pkl"
    with open(os.path.join(results_studies_dir, filename), 'rb') as f:
        return pickle.load(f)


def read_best_params_from_json_joint(model_short_name):
    """
    Read best parameters from a json file for joint optimization.
    
    Args:
        model_short_name: One of ('lr', 'et', 'xgb') representing the model.
    
    Returns:
        Dictionary of best parameters.
    """
    filename = f"{model_short_name}_best_params_joint.json"
    with open(os.path.join(results_bestparams_dir, filename), 'r') as f:
        return json.load(f)


def read_best_params_from_json_marginal(model_name, sampler_name, dataset_id):
    """
    Read best parameters from a json file for marginal optimization.
    
    Args:
        model_name: Name of the model. One of ('lr', 'et', 'xgb').
        sampler_name: Name of the sampler. One of ('RS', 'BS')
        dataset_id: ID of the dataset.
    
    Returns:
        Dictionary of best parameters.
    """
    filename = f"{model_name}_{sampler_name}_ID{dataset_id}_best_params_marginal.json"
    with open(os.path.join(results_bestparams_dir, filename), 'r') as f:
        return json.load(f)
