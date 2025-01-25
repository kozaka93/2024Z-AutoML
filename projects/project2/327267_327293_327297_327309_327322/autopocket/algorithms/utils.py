import json
import os

class ResultsReader():
    """
    A class for reading and loading results from JSON files in a specified directory.
    This class handles the reading of multiple JSON result files from a given directory path,
    where each file contains results for different wrapper implementations. The results are
    stored in a dictionary with wrapper names as keys.

    Parameters
    ----------
    path : str
        The directory path containing the JSON result files.

    Attributes
    ----------
    path : str
        The directory path containing the result files.
    results : dict
        A dictionary containing the loaded results, where keys are wrapper names
        and values are the corresponding JSON content.

    Methods
    -------
    load_results()
        Loads all JSON result files from the specified directory into a dictionary.

    Example
    -------
    >>> reader = ResultsReader("/path/to/results")
    >>> results = reader.results
    >>> print(results['wrapper_name'])
    
    Notes
    -----
    - Result files should be in JSON format
    - File names should follow the pattern: '{wrapper_name}_results.json'
    - The '_results' suffix will be removed from the file name when used as dictionary key
    """
    def __init__(self, path):
        self.path = path
        self.results = self.load_results()
    
    def load_results(self):
        results_dir = self.path
        results = {}
        for file in os.listdir(results_dir):
            with open(os.path.join(results_dir, file), 'r') as f:
                wrapper_name = os.path.splitext(file)[0].replace('_results', '')
                results[wrapper_name] = json.load(f)
        return results