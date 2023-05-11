# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:35:15 2023

@author: Utilisateur
"""
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
from tensorflow.keras.models import load_model

# import cv2
import pickle
from sklearn.model_selection import train_test_split

SIZE=160
SEED=31

model = load_model('model_.h5')

with open('val_images' + str(SIZE)+'.pickle', "rb") as f:
    val_images = pickle.load(f)
    
with open('val_labels' + str(SIZE)+'.pickle', "rb") as f:
    val_labels = pickle.load(f)

# score = model.evaluate(val_images,
#                        val_labels)

pred = model.predict(np.array([val_images[2555]]))

predicted_class = np.argmax(pred)

print (predicted_class)