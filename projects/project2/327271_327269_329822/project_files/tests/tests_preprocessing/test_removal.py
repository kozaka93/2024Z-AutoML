import unittest
import pandas as pd
from medaid.preprocessing.column_removal import ColumnRemover

class TestColumnRemover(unittest.TestCase):
    def setUp(self):
        # Test DataFrame
        self.data = {
            'id': [1, 2, 3, 4, 5],  # Column with 'id', should be removed
            'name': ['Alice', 'Bob', 'Charlie', 'Max', 'Loui'],  # Long text, unique, might be removed
            'age': [25, 30, 35, 29, 24],  # Numeric column
            'salary': [50000, 60000, 70000, 80000, 90000],  # Numeric column
            'random_text': ['aaa', 'bbb', 'ccc', 'ddd', 'eee'],  # Text column with high uniqueness
            'duplicate_salary': [50000, 60000, 70000, 80000, 90000],  # Correlated with 'salary', should be removed
        }
        self.df = pd.DataFrame(self.data)
        self.target_column = 'salary'
        self.remover = ColumnRemover(threshold=0.5, target_column=self.target_column)

    def test_remove_id_columns(self):
        cleaned_df = self.remover.remove(self.df)
        print(cleaned_df)
        self.assertNotIn('id', cleaned_df.columns, "Column 'id' was not removed.")

    def test_remove_high_unique_text_columns(self):
        cleaned_df = self.remover.remove(self.df)
        print(cleaned_df)
        self.assertNotIn('random_text', cleaned_df.columns, "Column 'random_text' was not removed.")

    def test_remove_highly_correlated_columns(self):
        cleaned_df = self.remover.remove(self.df)
        print(cleaned_df)
        self.assertNotIn('duplicate_salary', cleaned_df.columns, "Column 'duplicate_salary' was not removed.")

    def test_keep_valid_columns(self):
        cleaned_df = self.remover.remove(self.df)
        print(cleaned_df)
        self.assertIn('age', cleaned_df.columns, "Column 'age' was incorrectly removed.")
        self.assertIn('salary', cleaned_df.columns, "Column 'salary' was incorrectly removed.")

    def test_removal_info(self):
        self.remover.remove(self.df)
        removal_info = self.remover.get_removal_info()
        print(removal_info)
        self.assertTrue(removal_info['id']['Removed'], "Removal info for 'id' is incorrect.")
        self.assertTrue(removal_info['random_text']['Removed'], "Removal info for 'random_text' is incorrect.")
        self.assertTrue(removal_info['duplicate_salary']['Removed'], "Removal info for 'duplicate_salary' is incorrect.")
        self.assertFalse(removal_info['age']['Removed'], "Removal info for 'age' is incorrect.")
        self.assertFalse(removal_info['salary']['Removed'], "Removal info for 'salary' is incorrect.")

if __name__ == "__main__":
    unittest.main(buffer=True)
