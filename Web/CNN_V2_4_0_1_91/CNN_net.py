# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:31:46 2023

@author: arscg
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, backend, layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

SIZE=160
BATCH_SIZE = 15
SEED = 30

import pathlib

path=pathlib.Path(__file__).parent.absolute()
path = str(path.parent) + "\dataset_" + str(SIZE) + ".pickle"

print(path)

# Ouverture des données avec pickl
with open(path, "rb") as f:
    data = pickle.load(f)
    
df = pd.DataFrame.from_dict(data)

train_df, test_df = train_test_split(df, 
                                     test_size=0.3, 
                                     stratify=df['labels'], 
                                     random_state=SEED)

le = LabelEncoder()

train_images = np.array(train_df['images'].to_list())#/ 255.0
train_labels = le.fit_transform(np.array(train_df['labels'].to_list()))

test_images = np.array(test_df['images'].to_list())#/ 255.0
test_labels = le.fit_transform(np.array(test_df['labels'].to_list()))

model = models.Sequential()
model.add(layers.Conv2D(65, (2, 2), activation="relu", input_shape=df.iloc[0][0].shape))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (2, 2), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(256, (2, 2), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(512, (2, 2), activation="sigmoid"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(880, (2, 2), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(880, (2, 2), activation="relu"))
model.add(layers.MaxPool2D((3, 3)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.summary()

model.compile(optimizer="adamax",#adamax,
              loss="SparseCategoricalCrossentropy",
              metrics=["accuracy"])

history = model.fit(train_images,
                    train_labels,
                    validation_split = 0.2,
                    epochs = 25,
                    batch_size= BATCH_SIZE)

model.save('model_.h5')


with open('history.pickle', 'wb') as f:
    pickle.dump(history.history, f)
    
#vidage de la mémoire video du GPU
backend.clear_session()
    
model.evaluate(test_images,
               test_labels,
               batch_size= BATCH_SIZE)
    

#vidage de la mémoire video du GPU
backend.clear_session()

