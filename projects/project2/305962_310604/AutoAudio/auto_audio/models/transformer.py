from .base_model import AutoAudioBaseModel
import evaluate
import torch
from transformers import (
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    AutoFeatureExtractor,
)
import pandas as pd
import numpy as np
import uuid
from datasets import Dataset
from transformers import set_seed


# https://huggingface.co/docs/transformers/en/tasks/audio_classification
class AudioTransformer(AutoAudioBaseModel):
    def __init__(
        self, num_labels: int, label2id: dict, id2label: dict, random_state: int
    ):
        set_seed(random_state)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.label2id = label2id
        self.id2label = id2label
        self.model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
        self.id = str(uuid.uuid4())
        self.path = "outputs/transformer" + self.id

    def fit_from_audio(self, audios: pd.DataFrame):
        encoded_train_dataset = self.encode_dataset(audios)

        training_args = TrainingArguments(
            output_dir=self.path,
            eval_strategy="no",
            save_strategy="no",
            learning_rate=3e-5,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )

        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return accuracy.compute(
                predictions=predictions, references=eval_pred.label_ids
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_train_dataset,
            processing_class=self.feature_extractor,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    def encode_dataset(self, df):
        df["label"] = df["label"].map(self.label2id)
        dataset = Dataset.from_pandas(df)

        def preprocess_function(examples):
            audio_arrays = examples["audio"]
            inputs = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.feature_extractor.sampling_rate,
                max_length=16000,
                truncation=True,
            )
            return inputs

        encoded_dataset = dataset.map(
            preprocess_function, remove_columns="audio", batched=True
        )
        return encoded_dataset

    def predict_from_audio(self, audios: pd.DataFrame) -> np.ndarray:
        predictions = []
        with torch.no_grad():
            for audio in audios["audio"]:
                inputs = self.feature_extractor(
                    audio, sampling_rate=16000, return_tensors="pt"
                )
                logits = self.model(**inputs).logits
                predicted_class_ids = torch.argmax(logits).item()
                predicted_label = self.model.config.id2label[predicted_class_ids]
                predictions.append(predicted_label)
        return np.array(predictions)

    def get_data_type(self) -> str:
        return "audio"

    def __str__(self) -> str:
        return "Transformer"
