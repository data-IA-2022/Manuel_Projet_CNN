# -*- coding: utf-8 -*-
"""
Created on Fri May 19 08:19:02 2023

@author: Utilisateur
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# Répertoire source contenant les images originales
source_dir = './Dataset/PetImages 3 class/Horses/'

# Répertoire de destination pour sauvegarder les images augmentées
save_dir = './Dataset/PetImages 3 class augmente/Horses/'
os.makedirs(save_dir, exist_ok=True)

# Dimensions maximales pour le redimensionnement aléatoire
max_width = 300
max_height = 300

# Appel de ImageDataGenerator pour créer un générateur d'augmentation de données.
datagen = ImageDataGenerator(rotation_range=90,
                             horizontal_flip=True,
                             brightness_range=[0.2,1.0],
                             zoom_range=[0.5,1.0],
                             width_shift_range=[-20,20],
                             height_shift_range=0.5)

k = 0

# Boucle sur chaque fichier d'image dans le répertoire source
for filename in os.listdir(source_dir):
    print(filename)
    
    # Chargement de l'image
    img = load_img(os.path.join(source_dir, filename))
    
    # Conversion de l'image en tableau
    data = img_to_array(img)
    
    # Expansion de la dimension pour un seul échantillon
    samples = np.expand_dims(img, 0)

    # Création d'un itérateur pour l'augmentation de données
    it = datagen.flow(samples, batch_size=1)

    # Application de l'augmentation de données et sauvegarde des images augmentées
    for i in range(10):
        k += 1
        # Génération des images par lots
        batch = it.next()

        # N'oubliez pas de convertir ces images en entiers non signés pour les afficher
        image = batch[0].astype('uint8')

        # Sauvegarde de l'image augmentée
        save_path = os.path.join(save_dir, f'augmented_{k}.jpg')
        pyplot.imsave(save_path, image)
