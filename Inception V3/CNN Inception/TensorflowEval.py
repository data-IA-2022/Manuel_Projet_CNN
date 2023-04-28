# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:32:38 2022

@author: Utilisateur
"""

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle

def initialize():
    warnings.filterwarnings('ignore')
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    SIZE=160
    
    with open("dataset_" + str(SIZE)+".pickle", "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame.from_dict(data)

    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['labels'])

    le = LabelEncoder()

    train_images = np.array(train_df['images'].to_list())#/ 255.0
    train_labels = le.fit_transform(np.array(train_df['labels'].to_list()))

    test_images = np.array(test_df['images'].to_list())#/ 255.0
    test_labels = le.fit_transform(np.array(test_df['labels'].to_list()))    

    model = InceptionV3 (include_top=True,
                      weights=None,
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
                        batch_size= 40)

    return history
 
print(tf.test.is_gpu_available())

model,train_images, train_labels, test_images, test_labels = initialize()
history = process(model, train_images, train_labels, test_images, test_labels)
model.save('model_inceptionV3.h5')

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylim(0.5, 1)
plt.yticks(np.arange(0.5, 1.1, 0.1))
plt.legend(["training_"+ str(np.max(train_labels+1)), "test"])
plt.show()

model.evaluate(test_images, test_labels)

#vidage de la mémoire video du GPU
tf.keras.backend.clear_session()



