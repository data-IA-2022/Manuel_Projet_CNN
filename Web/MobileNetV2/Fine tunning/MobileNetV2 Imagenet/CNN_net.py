# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:32:38 2022

@author: Utilisateur
"""

import tensorflow as tf
# from tensorflow.keras.applications.resnet import ResNet50 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

BATCH_SIZE = 15
SIZE=160

def initialize():
    
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
                                         random_state=30)

    le = LabelEncoder()

    train_images = np.array(train_df['images'].to_list())#/ 255.0
    train_labels = le.fit_transform(np.array(train_df['labels'].to_list()))

    test_images = np.array(test_df['images'].to_list())#/ 255.0
    test_labels = le.fit_transform(np.array(test_df['labels'].to_list()))    

    model=MobileNetV2(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=(SIZE,SIZE,3),
                      pooling='max',
                      classes=np.max(train_labels+1))

    print("Model")

    optimizer="adamax"

    model.compile(optimizer = optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.summary()

    
    return model, train_images, train_labels, test_images, test_labels 
    
def process(model, train_images, train_labels, test_images, test_labels):   
    print('Fit')

    history = model.fit(train_images, 
                        train_labels,
                        validation_split = 0.2,
                        epochs = 25,
                        batch_size= BATCH_SIZE)

    return history
 
print(tf.test.is_gpu_available())

model,train_images, train_labels, test_images, test_labels = initialize()
history = process(model, train_images, train_labels, test_images, test_labels)
model.save('model_MobileNetV2.h5')

with open('history.pickle', 'wb') as f:
    pickle.dump(history.history, f)

#vidage de la mémoire video du GPU
tf.keras.backend.clear_session()

# Import du modèle sauvegardé
from tensorflow.keras.models import load_model

model = load_model('model_MobileNetV2.h5')
    
model.evaluate(test_images,
               test_labels,
               batch_size= BATCH_SIZE)

#vidage de la mémoire video du GPU
tf.keras.backend.clear_session()



