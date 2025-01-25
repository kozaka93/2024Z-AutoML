from abc import ABC, abstractmethod

class AbstractModelTrainer(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass