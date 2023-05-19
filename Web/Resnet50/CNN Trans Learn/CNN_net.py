# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:04:31 2023

@author: Utilisateur
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SIZE=160
BATCH_SIZE = 15
SEED = 30

import pathlib

path=pathlib.Path(__file__).parent.absolute()
path = str(path.parent.parent) + "\dataset_" + str(SIZE) + ".pickle"

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

xx_train = np.array(train_df['images'].to_list())#/ 255.0
yy_train = le.fit_transform(np.array(train_df['labels'].to_list()))

xx_test = np.array(test_df['images'].to_list())#/ 255.0
yy_test = le.fit_transform(np.array(test_df['labels'].to_list()))    

model = Sequential()

model.add(ResNet50 (include_top=False,
                  weights="imagenet",
                  pooling='max'))

# model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation = 'softmax'))
model.layers[0].trainable = False

model.summary()

optimizer="adamax"
 
model.compile(optimizer = optimizer,
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(xx_train, 
                    yy_train,
                    validation_split = 0.2,
                    epochs = 25,
                    batch_size= BATCH_SIZE)

model.save('model_.h5')

with open('history.pickle', 'wb') as f:
    pickle.dump(history.history, f)

#vidage de la mémoire video du GPU
clear_session()

# Import du modèle sauvegardé
from tensorflow.keras.models import load_model

model = load_model('model_.h5')

model.evaluate(xx_test,
               yy_test,
               batch_size= BATCH_SIZE)

#vidage de la mémoire video du GPU
clear_session()