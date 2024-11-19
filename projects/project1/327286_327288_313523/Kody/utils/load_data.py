from sklearn.preprocessing import LabelEncoder
import openml
def download_dataset(dataset_id):
    """
    Fetches datasets from OpenML based on provided dataset ID.

    Parameters:
    ----------
    dataset_ids : list of int
        List of OpenML dataset IDs to fetch.

    Returns:
    -------
    data : pd.DataFrame
        Loaded dataset as pandas DataFrames.
    """  
    
    data = openml.datasets.get_dataset(dataset_id).get_data(dataset_format='dataframe')[0]
    total_missing = data.isnull().sum().sum()
    print(f"Dataset ID={dataset_id}, shape: {data.shape}, {total_missing} missing values")

    return data


def get_data(dataset_ids):
    """
    Main function that loads datasets, preprocesses features and target,
    and splits data into training and testing sets.

    Parameters:
    ----------
    dataset_ids : list of int
        List of OpenML dataset IDs to fetch and preprocess.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before the split.

    Returns:
    -------
    X_trains, X_tests, y_trains, y_tests : list of np.ndarray
        Lists of training and testing sets for features and target.
    """
    le = LabelEncoder()
    Xs, ys = [], []

    for id in dataset_ids:
        dataset = download_dataset(id)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        y_encoded = le.fit_transform(y)

        Xs.append(X)
        ys.append(y_encoded)

    return Xs, ys