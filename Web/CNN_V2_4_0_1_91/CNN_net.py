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

with open("dataset_" + str(SIZE)+".pickle", "rb") as f:
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
                    validation_split = 0.1,
                    epochs = 25,
                    batch_size= BATCH_SIZE)

history.history.keys()

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylim(0.5, 1)
plt.yticks(np.arange(0.5, 1.1, 0.1))
plt.legend(["training_"+ str(np.max(train_labels+1)), "test"])
plt.savefig('Figure.jpeg',format='jpeg')
plt.show()


with open('history.pickle', 'wb') as f:
    pickle.dump(history.history, f)
    
model.evaluate(test_images,
               test_labels,
               batch_size= BATCH_SIZE)


model.save('model_.h5')

predict_list=[]
predict_reject_list=[]

for i in range(0, len(test_images)):
    pred = model.predict(np.array([test_images[i]]))
    predicted_class = np.argmax(pred)
    if i%500==0:
        print (i)
    predict_list.append(predicted_class)
    if predicted_class != test_labels[i]:
        predict_reject_list.append(i)

with open('predict_list.pickle', 'wb') as f:
    pickle.dump(predict_list, f)
    
with open('predict_reject_list.pickle', 'wb') as f:
    pickle.dump(predict_reject_list, f)
    
with open('test_images' + str(SIZE)+'.pickle', 'wb') as f:
    pickle.dump(test_images, f)

with open('test_labels' + str(SIZE)+'.pickle', 'wb') as f:
    pickle.dump(test_labels, f)
    

#vidage de la mémoire video du GPU
backend.clear_session()

