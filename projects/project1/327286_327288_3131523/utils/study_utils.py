import numpy as np
import optuna


def sample_parameter(trial, param_name, value):
    """Sample a parameter value based on its distribution type."""
    if len(value) == 3:
        low, high, dist_type = value
        if dist_type == 'log':
            return trial.suggest_float(param_name, low, high, log=True)
        if dist_type == 'float':
            return trial.suggest_float(param_name, low, high)
        else:
            return trial.suggest_int(param_name, low, high)
    elif len(value) == 2:
        options, dist_type = value
        return trial.suggest_categorical(param_name, options)
    else:
        raise ValueError('Invalid parameter value')

def get_trials(study):
    trials = (
        study.trials_dataframe(attrs=("number", "params", "value"))
        .rename(columns={'values_0': 'AUC', 'values_1': 'Accuracy', 'values_2': 'F1 Score', 'number': 'trial'})
        .set_index('trial')
    )
    trials.columns = trials.columns.str.replace('params_', '')

    # Also get the time of each trial datetime_complete - datetime_start from attrs
    time = (study.trials_dataframe(attrs=("number", "datetime_start", "datetime_complete"))
                      .rename(columns={'number': 'trial'})
                      .set_index('trial'))
    
    time['time'] = (time['datetime_complete'] - time['datetime_start']).dt.total_seconds()
    time = time.drop(columns=['datetime_start', 'datetime_complete'])
    trials = trials.join(time)

    return trials
