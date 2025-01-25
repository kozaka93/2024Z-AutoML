#!/usr/bin/env python
# coding: utf-8

# ### Description
# This Notebook shows how to train and use the library for music genre classification.

# ### Download data
# For this part you will need to have kaggle installed. `pip install kaggle`.
# Alternativaly you could download the dataset from `https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification` and manually extract it into `../data/gtzan-dataset-music-genre-classification`

# In[1]:


import os
import subprocess

dataset_path = "data/gtzan-dataset-music-genre-classification"

if not os.path.exists(dataset_path):
    print("Dataset not found. Downloading...")
    os.makedirs(dataset_path, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "andradaolteanu/gtzan-dataset-music-genre-classification",
            "-p",
            dataset_path,
            "--unzip",
        ]
    )
    print("Download complete.")
else:
    print("Dataset already exists.")


# ### Data format
# We first have to create a dataframe which stores all the data (file paths and labels)

# In[2]:


import pandas as pd


# In[3]:


genres_path = os.path.join(dataset_path, "Data/genres_original")
paths = []
labels = []
for genre in os.listdir(genres_path):
    folder_path = os.path.join(genres_path, genre)
    for filename in os.listdir(folder_path):
        paths.append(os.path.join(folder_path, filename))
        labels.append(genre)
df = pd.DataFrame({"file_path": paths, "label": labels})
df.sample(5)


# Import library

# In[4]:


import sys
import os

sys.path.insert(0, os.path.abspath("../src"))


# ### Model training

# In[5]:


from auto_audio_model import AutoAudioModel

df_train = df.sample(200, random_state=42)
df_test = df.sample(100, random_state=42)
model = AutoAudioModel()
model.fit(df_train, time_limit=500)


# In[6]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

y_test = df_test["label"]
y_pred = model.predict(df_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot()
plt.xticks(rotation=90)
plt.show()
