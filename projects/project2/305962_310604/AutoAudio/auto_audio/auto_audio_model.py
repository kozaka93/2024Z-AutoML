import copy
from typing import Optional
import pandas as pd
import numpy as np
from .hyperparameter_tuner import HyperparameterTuner
from .models.base_model import AutoAudioBaseModel
import auto_audio.preprocessing as pre
from .models.svm import AudioSVM
from .models.knn import AudioKNN
from .models.xgb import AudioGB
from .models.transformer import AudioTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from auto_audio.stopwatch import Stopwatch


class AutoAudioModel:
    def __init__(self, log=True):
        self.models = []
        self.best_model = None
        self.timings = {}
        self.info = {}
        self.logging_enabled = log
        self._is_test = False
        self._train_transformer = torch.cuda.is_available()

    def fit(self, data: pd.DataFrame, time_limit: int, tuner: Optional[HyperparameterTuner] = None, random_state: int = 42):
        """
        Fit the model to the training data.

        Parameters:
        data (pd.DataFrame): A DataFrame containing two columns:
            - 'file_path': The full path to the audio file.
            - 'label': The label associated with the audio file.
        tuner (HyperparameterTuner): The tuner to use for hyperparameter optimization.
        time_limit (int): The time limit for training the model (in seconds).
        random_state (int): The random state for reproducibility.

        Raises:
        ValueError: If the DataFrame does not contain the required columns.
        """

        sw = Stopwatch.start_new()
        if not {"file_path", "label"}.issubset(data.columns):
            raise ValueError("DataFrame must contain 'file_path' and 'label' columns")

        if not self._is_time_limit_enough(data, time_limit, random_state):
            self.log(
                "Not enough time to train model. Please increase the time limit or cut the dataset."
            )
            return

        dataset = self._prepare_datesets(data, random_state)
        if sw.elapsed_time() > time_limit:
            self.log(
                "Not enough time to train model. Please increase the time limit or cut the dataset."
            )
            return
        self.models = self._get_models(dataset["labels_train"], random_state)
        self._train_models(dataset, int(time_limit - sw.elapsed_time()), tuner)

        self.timings["total"] = sw.elapsed_time()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on the test data.

        Parameters:
        data (pd.DataFrame): A DataFrame containing one column:
            - 'file_path': The full path to the audio file.

        Returns:
        np.ndarray: The predicted labels.

        Raises:
        ValueError: If the model has not been trained or if the DataFrame does not contain the required column.
        """

        if self.best_model is None:
            raise ValueError("Model has not been trained")
        if "file_path" not in data.columns:
            raise ValueError("DataFrame must contain 'file_path' column")
        features, audios, _ = pre.aggregate_audio_features(data)
        if self.best_model.get_data_type() == "audio":
            return self.best_model.predict(audios)
        return self.best_model.predict(features)

    def _is_time_limit_enough(self, data, time_limit, random_state) -> bool:
        if self._is_test:
            return True
        n_samples = max(10, int(len(data) * 0.01))
        coef = int(len(data)) / n_samples
        data = data.sample(n=n_samples, random_state=random_state)
        ghost_model = AutoAudioModel(log=False)
        ghost_model._is_test = True
        ghost_model.fit(data, time_limit, None, random_state)
        if ghost_model.timings["total"] > time_limit / coef:
            if "Transformer" in ghost_model.timings["model_training"] and (
                ghost_model.timings["total"]
                - ghost_model.timings["model_training"]["Transformer"]
                > time_limit / coef
            ):
                self._train_transformer = False
                return True
            return False
        return True

    def _prepare_datesets(
        self, data: pd.DataFrame, random_state: int
    ) -> dict[str, pd.DataFrame]:
        sw = Stopwatch.start_new()

        data.reset_index(drop=True, inplace=True)
        self.log("Preprocessing audio files.")
        features, audios, labels = pre.aggregate_audio_features(data)
        self.log("Finished preprocessing files.")
        features.reset_index(drop=True, inplace=True)
        audios.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)

        test_size = 0.2
        indices = labels.index
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, shuffle=True
        )
        features_train = features.loc[train_indices]
        labels_train = labels.loc[train_indices].values.reshape(-1)
        audios_train = audios.loc[train_indices]
        features_test = features.loc[test_indices]
        labels_test = labels.loc[test_indices].values.reshape(-1)
        audios_test = audios.loc[test_indices]

        self.timings["data_preparation"] = sw.elapsed_time()

        return {
            "features_train": features_train,
            "labels_train": labels_train,
            "audios_train": audios_train,
            "features_test": features_test,
            "labels_test": labels_test,
            "audios_test": audios_test,
        }

    def _get_models(self, labels, random_state) -> list[AutoAudioBaseModel]:
        unique = np.unique(labels)
        n_unique = len(unique)

        label2id = {}
        id2label = {}
        for i, label in enumerate(unique):
            label2id[label] = i
            id2label[i] = label
        models = [
            AudioSVM(random_state),
            AudioKNN(n_unique),
            AudioGB(random_state),
        ]
        if self._train_transformer:
            models.append(AudioTransformer(n_unique, label2id, id2label, random_state))
        else:
            self.log("Cuda not available. Not training transformer model.")
        return models

    def _train_models(self, dataset: dict[str, pd.DataFrame], time_limit: int, tuner: Optional[HyperparameterTuner]):
        self.timings["model_training"] = {}
        best_accuracy = -1
        total_sw = Stopwatch.start_new()
        for model in self.models:
            best_model_of_type_accuracy = -1
            best_model_of_type = None
            sw = Stopwatch.start_new()
            self.log(f"Training {model}")
            self.info["model_accuracies"] = {}
            if model.get_data_type() == "audio":
                model.fit_from_audio(dataset["audios_train"])
                predictions = model.predict_from_audio(dataset["audios_test"])
            else:
                model.fit(dataset["features_train"], dataset["labels_train"])
                predictions = model.predict(dataset["features_test"])
            accuracy = accuracy_score(dataset["labels_test"], predictions)
            self.info["model_accuracies"][str(model)] = accuracy
            self.log(f"{model} achieved {round(accuracy * 100, 1)}% accuracy.")
            self.timings["model_training"][str(model)] = sw.elapsed_time()
            if tuner is not None and model is not AudioTransformer:
                self.log("Tuning model hyperparameters.")
                tuned_estimator = tuner.tune(
                    model, dataset["features_train"], dataset["labels_train"]
                )
                tuned_model = copy.deepcopy(model)
                tuned_model.set_params(**tuned_estimator.get_params())
                tuned_model.fit(dataset["features_train"], dataset["labels_train"])
                    
                tuned_predictions = tuned_model.predict(dataset["features_test"])
                tuned_accuracy = accuracy_score(dataset["labels_test"], tuned_predictions)
                self.log(f"Tuned {model} achieved {round(tuned_accuracy * 100, 1)}% accuracy.")
                    
                if tuned_accuracy > accuracy:
                    best_model_of_type = tuned_model
                    best_model_of_type_accuracy = tuned_accuracy
                else:
                    self.log(f"Tuning did not improve {model}. Keeping original.")
                    best_model_of_type = model
                    best_model_of_type_accuracy = accuracy
            else:
                best_model_of_type = model
                best_model_of_type_accuracy = accuracy
                
            if best_model_of_type_accuracy > best_accuracy:
                self.best_model = best_model_of_type
                best_accuracy = best_model_of_type_accuracy
                self.info["best_accuracy"] = best_model_of_type_accuracy

            if total_sw.elapsed_time() > time_limit:
                self.log("Not enough time to train all models. Stopping now.")
                break

        self.log("Finished training.")
        self.log(f"Best model is: {str(self.best_model)}")

    def log(self, message: str):
        if self.logging_enabled:
            print(message)
