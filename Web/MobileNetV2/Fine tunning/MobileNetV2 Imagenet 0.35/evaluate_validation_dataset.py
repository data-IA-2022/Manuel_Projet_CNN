# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:57:46 2023

@author: Utilisateur
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# Paramètres
SIZE=160
SEED=30

import pathlib

path=pathlib.Path(__file__).parent.absolute()
path = str(path.parent.parent.parent) + "\dataset_" + str(SIZE) + ".pickle"

print(path)

# Ouverture des données avec pickl
with open(path, "rb") as f:
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

model = load_model('model_MobileNetV2.h5')

model.summary()

 
# Encodage des étiquettes avec LabelEncoder    
le = LabelEncoder()
  
# Conversion des images et des étiquettes en tableaux Numpy
val_images = np.array(test_df['images'].to_list())#/ 255.0
val_labels = le.fit_transform(np.array(test_df['labels'].to_list()))    

# Évaluation du modèle sur les données de validation
pred_labels = np.argmax(model.predict(val_images), axis=-1)
confusion_mtx = confusion_matrix(val_labels, pred_labels)

# Sauvegarde de la matrice de confusion avec pickle
with open('confusion_matrix.pickle', 'wb') as f:
    pickle.dump(confusion_mtx, f)

# Évaluation du modèle sur les données de validation
score = model.evaluate(val_images,
                       val_labels)

# Calcul du F-score
precision, recall, f_score, _ = precision_recall_fscore_support(val_labels, pred_labels, average='weighted')


performance = [precision, recall, f_score]

# Sauvegarde du score et du F-score
with open('score.pickle', 'wb') as f:
    pickle.dump(score, f)

with open('performance.pickle', 'wb') as f:
    pickle.dump(performance, f)

# Affichage du score et du F-score
print("---------------------------------------------------------")
print(f"Score : {score}")
print(f"F-score : {f_score}")
print("---------------------------------------------------------")

      
# Affichage de la matrice de confusion
plt.figure(figsize=(8,8))
plt.imshow(confusion_mtx, cmap=plt.cm.Blues)
plt.title('Matrice de confusion')
plt.colorbar()
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45)
plt.yticks(tick_marks, le.classes_)
plt.xlabel('Prédictions')
plt.ylabel('Valeurs réelles')
plt.tight_layout()
plt.show() 

