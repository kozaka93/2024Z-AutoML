from abc import ABC, abstractmethod
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import cloudpickle

class AbstractModelManager(ABC):
    """
    Abstract base class for managing scikit-learn models.
    """

    @abstractmethod
    def get_models(self):
        """
        Initialize and return a dictionary of scikit-learn models.
        """
        pass

    @abstractmethod
    def get_model(self, name):
        """
        Retrieve a specific model by name.
        
        Parameters:
            name (str): The name of the model to retrieve.
        
        Returns:
            model: The scikit-learn model instance.
        """
        pass

    @abstractmethod
    def save_models(self):
        """
        Save all models to a pickle file.
        """
        pass

    @abstractmethod
    def load_models(self):
        """
        Load models from a pickle file.
        
        """
        pass


    def save_model(model, filename: str):
        with open(filename, "wb") as f:
            cloudpickle.dump(model, f)

    def load_model(filename: str):
        with open(filename, "rb") as f:
            return cloudpickle.load(f)

class ModelManager(AbstractModelManager):
    """
    Concrete implementation of AbstractModelManager.
    """

    def __init__(self):
        self.models = {}
        self.get_models()  # Initialize models upon creation

    def get_models(self):
        """
        Initialize and return a dictionary of scikit-learn models.
        """
        if not self.models:
            self.models = {
                'logistic_regression': LogisticRegression(),
                'decision_tree': DecisionTreeClassifier(),
                'svc': SVC()
            }
        return self.models

    def get_model(self, name):
        """
        Retrieve a specific model by name.
        """
        model = self.models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self.models.keys())}")
        return model

    def save_models(self, filepath):
        """
        Save all models to a pickle file.
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.models, f)
            print(f"Models successfully saved to {filepath}.")
        except Exception as e:
            print(f"An error occurred while saving models: {e}")

    def load_models(self, filepath):
        """
        Load models from a pickle file.
        """
        try:
            with open(filepath, 'rb') as f:
                self.models = pickle.load(f)
            print(f"Models successfully loaded from {filepath}.")
        except FileNotFoundError:
            print(f"The file {filepath} does not exist.")
        except Exception as e:
            print(f"An error occurred while loading models: {e}")
