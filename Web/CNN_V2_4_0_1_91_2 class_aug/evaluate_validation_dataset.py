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
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend
from sklearn.preprocessing import LabelBinarizer

# Paramètres
SIZE=160
SEED=30

import pathlib

path=pathlib.Path(__file__).parent.absolute()
path = str(path.parent) + "\dataset_3_" + str(SIZE) + ".pickle"

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

model = load_model('model_.h5')
 
lb = LabelBinarizer()
val_labels_one_hot = lb.fit_transform(np.array(test_df['labels'].to_list()))  

# Conversion des images et des étiquettes en tableaux Numpy
val_images = np.array(test_df['images'].to_list())#/ 255.0


# Calcul de la courbe ROC pour chaque classe
fpr = dict()
tpr = dict()
thresholds = dict()
n_classes = val_labels_one_hot.shape[1]

try:
    pred_labels = model.predict_classes(val_images)
except:
    pred_labels = model.predict(val_images)
    
auc_score=[]

try:
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(val_labels_one_hot[:, i], pred_labels)
        auc_score.append(auc(fpr[i], tpr[i]))
except:
    pred_labels = pred_labels.argmax(axis=-1)
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(val_labels_one_hot[:, i], pred_labels)
        auc_score.append(auc(fpr[i], tpr[i]))

confusion_mtx = confusion_matrix(np.argmax(val_labels_one_hot, axis=1), pred_labels)

# Sauvegarde de la matrice de confusion avec pickle
with open('confusion_matrix.pickle', 'wb') as f:
    pickle.dump(confusion_mtx, f)

with open('roc_fpr.pickle', 'wb') as f:
    pickle.dump(fpr, f)
    
with open('roc_tpr.pickle', 'wb') as f:
    pickle.dump(tpr, f)

# Évaluation du modèle sur les données de validation
score = model.evaluate(val_images,
                        np.argmax(val_labels_one_hot, axis=1))

# Calcul du F-score
precision, recall, f_score, _ = precision_recall_fscore_support(np.argmax(val_labels_one_hot, axis=1), pred_labels, average='weighted')


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
tick_marks = np.arange(len(lb.classes_))
plt.xticks(tick_marks, lb.classes_, rotation=45)
plt.yticks(tick_marks, lb.classes_)
plt.xlabel('Prédictions')
plt.ylabel('Valeurs réelles')
plt.tight_layout()
plt.show() 

for i in range(n_classes):
    plt.figure(figsize=(8, 8))
    plt.plot(fpr[i], tpr[i], label='Courbe ROC ' + str(round(auc_score[i], 3)))
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend()
    plt.show()

plt.figure(figsize=(8, 8))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {auc(fpr[i], tpr[i]):.3f})')

plt.plot([0, 1], [0, 1], 'k--')  # Ligne en pointillés représentant le classifieur aléatoire
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC')
plt.legend(loc='lower right')
plt.show()


#vidage de la mémoire video du GPU
backend.clear_session()

