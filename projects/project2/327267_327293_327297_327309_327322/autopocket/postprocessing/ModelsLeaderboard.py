import json
import pandas as pd
import os

from autopocket.algorithms.utils import ResultsReader

class ModelsLeaderboard:
    
    def __init__(self, results_reader: ResultsReader): ####
        """
        Class for creating a leaderboard of models.
        """
        self.reader = results_reader ####

    def create_leaderboard(self): ####
        """
        Generates leaderboard based on the results of the best model searching process.
        """
        try:
            res = self.reader.results ####

            leaderboard_data = []
            for model_name, model_data in res.items():
                score = model_data['score']
                params = model_data['params']
                
                model_row = { 
                    'model_name': model_name,
                    'score': score,
                    'params': params
                }
                leaderboard_data.append(model_row)

            leaderboard = pd.DataFrame(leaderboard_data)
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 2000)
            pd.set_option('display.max_colwidth', None)
            
            leaderboard = leaderboard.sort_values(by='score', ascending=False)
            leaderboard.insert(0, 'position', range(1, len(leaderboard) + 1))
            leaderboard = leaderboard.reset_index(drop=True)
            print(leaderboard.to_string(index=False))
            
            return leaderboard
        except Exception as e:
            print(f"Error while creating leaderboard: {e}")
            return None
        
    def save_leaderboard_to_csv(self, leaderboard: pd.DataFrame, results_dir: str): ####
        """
        Saves the leaderboard to a csv file.
        
        Parameters:
        leaderboard: pd.DataFrame
            The leaderboard to save.
        results_dir: str
            The directory to save the leaderboard to.
        """
        try:
            os.makedirs(os.path.join(results_dir), exist_ok=True)
            results_dir = os.path.join(results_dir)

            leaderboard.to_csv(os.path.join(results_dir, 'leaderboard.csv'), index=False)
            
            print(f"Saving results to results/explanations")
        except Exception as e:
            print(f"Error while saving leaderboard to csv: {e}")
