# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:57:46 2023

@author: Utilisateur
"""
# import streamlit as st
import pandas as pd
from joblib import load
# import index

import datetime
# import streamlit as st
import plotly.express as px
import numpy as np
# import Accueil as index
import cv2
from sklearn.preprocessing import LabelEncoder


# import cv2
import pickle
from sklearn.model_selection import train_test_split

SIZE=160
SEED=31

with open("dataset_" + str(SIZE)+".pickle", "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame.from_dict(data)

train_df, test_df = train_test_split(df, 
                                     test_size=0.3,
                                     stratify=df['labels'],
                                     random_state=SEED)


from tensorflow.keras.models import load_model



model = load_model('model_.h5')
     
le = LabelEncoder()
  
val_images = np.array(test_df['images'].to_list())#/ 255.0
val_labels = le.fit_transform(np.array(test_df['labels'].to_list()))    

with open('val_images' + str(SIZE)+'.pickle', 'wb') as f:
    pickle.dump(val_images, f)

with open('val_labels' + str(SIZE)+'.pickle', 'wb') as f:
    pickle.dump(val_labels, f)

score = model.evaluate(val_images,
                       val_labels)