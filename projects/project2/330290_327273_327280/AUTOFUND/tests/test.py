from numerai_automl.utils.utils import get_project_root

print(get_project_root())


from numerai_automl.data_managers.data_loader import DataLoader

data_loader = DataLoader(data_version="v5.0", feature_set="all")



data = data_loader.get_features()

print(data)
