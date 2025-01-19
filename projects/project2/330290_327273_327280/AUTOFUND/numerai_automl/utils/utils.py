# numerai_automl/utils.py

import pandas as pd
import cloudpickle
import os

def save_model(model, filename: str):
    with open(filename, "wb") as f:
        cloudpickle.dump(model, f)

def load_model(filename: str):
    with open(filename, "rb") as f:
        return cloudpickle.load(f)


def get_project_root():
    """
    Get the root directory of the project
    {$HOME}/AUTOFUND
    """
    current_file_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

if __name__ == "__main__":
    print(get_project_root())
