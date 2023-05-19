# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:42:55 2023

@author: Utilisateur
"""
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, backend, layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

SIZE=160
BATCH_SIZE = 15
SEED = 30

import pathlib

path=pathlib.Path(__file__).parent.absolute()
path = str(path.parent.parent.parent) + "\dataset_" + str(SIZE) + ".pickle"

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

xx_train = np.array(train_df['images'].to_list())
yy_train = le.fit_transform(np.array(train_df['labels'].to_list()))

xx_test = np.array(test_df['images'].to_list())
yy_test = le.fit_transform(np.array(test_df['labels'].to_list()))

e = LabelEncoder()

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
model.add(layers.Dense(3, activation="softmax"))

model.summary()

print("Model")

optimizer="adamax"

# model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

datagen = ImageDataGenerator(#rescale=1/255.,
                             rotation_range=20,
                             zoom_range=0.15,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.15,
                             horizontal_flip=True,
                             validation_split=0.2)

training_generator = datagen.flow(xx_train, 
                                  yy_train, 
                                  batch_size=BATCH_SIZE,
                                  subset='training',
                                  seed=5)

validation_generator = datagen.flow(xx_train, 
                                    yy_train, 
                                    batch_size=BATCH_SIZE,
                                    subset='validation',
                                    seed=5)

history = model.fit_generator(training_generator,
                              steps_per_epoch=(len(xx_train)*0.8)//BATCH_SIZE, 
                              epochs=25, 
                              validation_data=validation_generator, 
                              validation_steps=(len(xx_train)*0.2)//BATCH_SIZE)

model.save('model_.h5')

with open('history.pickle', 'wb') as f:
    pickle.dump(history.history, f)

#vidage de la mémoire video du GPU
backend.clear_session()
    
# Import du modèle sauvegardé
from tensorflow.keras.models import load_model

model = load_model('model_.h5')

model.evaluate(xx_test,
               yy_test,
               batch_size= BATCH_SIZE)
    

#vidage de la mémoire video du GPU
backend.clear_session()


