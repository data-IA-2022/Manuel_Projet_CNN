# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:57:46 2023

@author: Utilisateur
"""
# import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split

# Paramètres
SIZE=160
SEED=31

# Ouverture des données avec pickl
with open("dataset_" + str(SIZE)+".pickle", "rb") as f:
    data = pickle.load(f)

# Conversion des données en DataFrame
df = pd.DataFrame.from_dict(data)

# Séparation des données en jeu d'entraînement et de test
train_df, test_df = train_test_split(df, 
                                     test_size=0.3,
                                     stratify=df['labels'],
                                     random_state=SEED)

# Import du modèle sauvegardé
from tensorflow.keras.models import load_model

model = load_model('model_.h5')
 
# Encodage des étiquettes avec LabelEncoder    
le = LabelEncoder()
  
# Conversion des images et des étiquettes en tableaux Numpy
val_images = np.array(test_df['images'].to_list())#/ 255.0
val_labels = le.fit_transform(np.array(test_df['labels'].to_list()))    

# Enregistrement des images de validation et des labels
with open('val_images' + str(SIZE)+'.pickle', 'wb') as f:
    pickle.dump(val_images, f)

with open('val_labels' + str(SIZE)+'.pickle', 'wb') as f:
    pickle.dump(val_labels, f)

# Évaluation du modèle sur les données de validation
score = model.evaluate(val_images,
                       val_labels,
                       batch_size= 20)

# Affichage du score
print()
print("---------------------------------------------------------")
print( f"Score : {score}")
print("---------------------------------------------------------")
print()
      
      