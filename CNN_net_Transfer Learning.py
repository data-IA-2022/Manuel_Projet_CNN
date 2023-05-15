# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:04:31 2023

@author: Utilisateur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SIZE=224
BATCH_SIZE = 5
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

train_images = np.array(train_df['images'].to_list())#/ 255.0
train_labels = le.fit_transform(np.array(train_df['labels'].to_list()))

test_images = np.array(test_df['images'].to_list())#/ 255.0
test_labels = le.fit_transform(np.array(test_df['labels'].to_list()))    

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

history = model.fit(train_images, 
                    train_labels,
                    validation_split = 0.2,
                    epochs = 25,
                    batch_size= BATCH_SIZE)

model.save('model_resnet50_Transfère_Learning.h5')


plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylim(0.5, 1)
plt.yticks(np.arange(0.5, 1.1, 0.1))
plt.legend(["training_"+ str(np.max(train_labels+1)), "test"])
plt.savefig('Figure.jpeg',format='jpeg')
plt.show()

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylim(0.9, 1)
plt.yticks(np.arange(0.9, 1.1, 0.1))
plt.legend(["training_"+ str(np.max(train_labels+1)), "test"])
plt.savefig('Figure2.jpeg',format='jpeg')
plt.show()

with open('history.pickle', 'wb') as f:
    pickle.dump(history.history, f)
    
model.evaluate(test_images,
               test_labels,
               batch_size= BATCH_SIZE)



#vidage de la mémoire video du GPU
clear_session()