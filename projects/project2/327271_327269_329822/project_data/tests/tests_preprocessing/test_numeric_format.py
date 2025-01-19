import unittest
import pandas as pd
from medaid.preprocessing.numeric_format_handler import NumericCommaHandler

class TestNumericCommaHandler(unittest.TestCase):

    def setUp(self):
        """
        This method will run before each test case.
        It initializes a sample DataFrame with values containing commas as decimal separators.
        """
        self.df = pd.DataFrame({
            'column1': ['1,23', '4,56', '7,89', '10,12'],
            'column2': ['1.23', '4.56', '7.89', '10.12'],
            'column3': ['apple', 'banana', 'cherry', 'date']
        })
        self.handler = NumericCommaHandler()

    def test_detect_columns_with_numeric_commas(self):
        """
        Tests if the handler correctly detects columns with numeric values using commas as decimal separators.
        """
        columns_with_commas = self.handler.detect_columns_with_numeric_commas(self.df)
        self.assertEqual(columns_with_commas, ['column1'])

    def test_convert_numeric_commas(self):
        """
        Tests if the handler correctly converts numeric values with commas to floats.
        """
        self.handler.detect_columns_with_numeric_commas(self.df)
        processed_df = self.handler.convert_numeric_commas(self.df)

        # Verify that the values in 'column1' have been converted to floats
        self.assertEqual(processed_df['column1'][0], 1.23)
        self.assertEqual(processed_df['column1'][1], 4.56)
        self.assertEqual(processed_df['column1'][2], 7.89)
        self.assertEqual(processed_df['column1'][3], 10.12)

    def test_handle_numeric_format(self):
        """
        Tests if the handler correctly processes the DataFrame by detecting and converting numeric values with commas.
        """
        processed_df = self.handler.handle_numeric_format(self.df)

        # Verify that 'column1' has been converted to floats
        self.assertEqual(processed_df['column1'][0], 1.23)
        self.assertEqual(processed_df['column1'][1], 4.56)
        self.assertEqual(processed_df['column1'][2], 7.89)
        self.assertEqual(processed_df['column1'][3], 10.12)

        # Verify that 'column2' and 'column3' remain unchanged
        self.assertEqual(processed_df['column2'][0], '1.23')
        self.assertEqual(processed_df['column3'][0], 'apple')

if __name__ == '__main__':
    unittest.main()
