import unittest
import pandas as pd
from medaid.preprocessing.preprocessing import Preprocessing

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """
        Set up a sample DataFrame for testing.
        """
        self.data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],  # Column to be removed
            'text': ['apple', 'banana', 'apple', 'orange', 'pear'],  # Text column
            'age': ['25', '30', '35', '40', 'NaN'],  # Numeric column with missing value as strings
            'weight': ['60,5', '56,0', '70,0', '65,5', '75,0'],  # Numeric column with commas
            'salary': ['50,000', '60,000', 'NaN', '80,000', '90,000'],  # Numeric column with commas
            'target': ['yes', 'no', 'yes', 'no', 'yes']  # Target column (categorical)
        })

        self.target_column = 'target'
        self.preprocessing = Preprocessing(self.target_column)

    def test_handle_numeric_format(self):
        """
        Test handling numeric formats in the main dataset.
        """
        processed_df = self.preprocessing.numeric_format_handler.handle_numeric_format(self.data)
        print(f"format\n {processed_df}")

        # Check if numeric columns with commas are properly converted
        expected_age = [25.0, 30.0, 35.0, 40.0, None]
        expected_weight = [60.5, 56.0, 70.0, 65.5, 75.0]
        expected_salary = [50000.0, 60000.0, None, 80000.0, 90000.0]

        self.assertTrue(processed_df['age'].tolist() == expected_age)
        self.assertTrue(processed_df['weight'].tolist() == expected_weight)
        self.assertTrue(processed_df['salary'].tolist() == expected_salary)

    def test_remove_text_columns(self, threshold=0.5):
        """
        Test removing text columns.
        """
        processed_df = self.preprocessing.text_column_remover.remove(self.data)
        print(f"text\n processed_df")
        self.assertNotIn('text', processed_df.columns)  # Check if the text column was removed
        self.assertNotIn('id', processed_df.columns)  # Check if the 'id' column was removed

    def test_imputation(self):
        """
        Test imputation of missing values.
        """
        processed_df = self.preprocessing.imputation.impute_missing_values(self.data)
        print(f"impute\n {processed_df}")
        self.assertFalse(processed_df['age'].isnull().any())  # Check if missing values in 'age' were filled
        self.assertFalse(processed_df['salary'].isnull().any())  # Check if missing values in 'salary' were filled

    def test_encoding(self):
        """
        Test encoding of categorical columns.
        """
        processed_df = self.preprocessing.encoder.encode(self.data)
        print(f"encode\n {processed_df}")
        self.assertIn('target', processed_df.columns)  # Check if the target column was encoded
        self.assertNotIn('text', processed_df.columns)  # Check if the text column was removed

    def test_scaling(self):
        """
        Test scaling of numeric columns.
        """
        processed_df = self.preprocessing.scaler.scale(self.data)
        print(f"scale\n {processed_df}")
        for col in ['age', 'salary']:
            self.assertTrue(processed_df[col].min() >= 0)  # Check if values are scaled
            self.assertTrue(processed_df[col].max() <= 1)

    def test_full_pipeline(self):
        """
        Test the full preprocessing pipeline.
        """
        processed_df = self.preprocessing.preprocess(self.data)
        print(f"full\n {processed_df}")
        # Check if 'id' and 'text' columns were removed
        self.assertNotIn('id', processed_df.columns)
        self.assertNotIn('text', processed_df.columns)

        # Check if missing values were filled
        self.assertFalse(processed_df.isnull().any().any())

        # Check if numeric data was scaled
        for col in ['age', 'salary']:
            self.assertTrue(processed_df[col].min() >= 0)
            self.assertTrue(processed_df[col].max() <= 1)

        # Check if categorical data was encoded
        self.assertIn('target', processed_df.columns)
        self.assertTrue(processed_df['target'].dtype in [int, float])

if __name__ == "__main__":
    unittest.main()
