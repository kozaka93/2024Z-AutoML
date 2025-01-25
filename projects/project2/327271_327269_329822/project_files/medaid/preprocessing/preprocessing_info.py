import pandas as pd

class PreprocessingCsv:
    """
    A class for exporting preprocessing details to a CSV file.
    """
    def __init__(self, path):
        self.path = path

    def export_to_csv(self, text_column_removal_info, imputation_info, encoding_info, scaling_info):
        """
        Exports preprocessing details (removal, imputation, encoding, scaling) to a CSV file.

        Parameters:
        - text_column_removal_info (dict): Information about text column removal.
        - imputation_info (dict): Information about missing value imputation.
        - encoding_info (dict): Information about variable encoding.
        - scaling_info (dict): Information about feature scaling.
        """
        all_columns = set(
            text_column_removal_info.keys()
        ).union(imputation_info.keys(), encoding_info.keys(), scaling_info.keys())
        
        columns_info = []
        for column in all_columns:
            removal_info = text_column_removal_info.get(column, {})
            imputation_details = imputation_info.get(column, {})
            encoding_details = encoding_info.get(column, {})
            scaling_details = scaling_info.get(column, {})


            columns_info.append({
                "Column Name": column,
                "Removed": removal_info.get("Removed", ""),  
                "Reason for Removal": removal_info.get("Reason", ""),  
                "Correlation with Target": imputation_details.get("Correlation", ""),
                "Imputation Method": imputation_details.get("Imputation Method", ""),
                "Encoded": encoding_details.get("Encoded", ""),
                "Encoding Method": encoding_details.get("Encoding Method", ""),
                "Label Encoding Mapping": str(encoding_details.get("Mapping", "")),  
                "Scaling Method": scaling_details.get("scaling_method", ""),
                "Scaling Params": str(scaling_details.get("params", "")),  
            })

        df = pd.DataFrame(columns_info)

        df.to_csv(self.path, index=False)
