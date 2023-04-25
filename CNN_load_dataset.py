# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:06:58 2023

@author: Utilisateur
"""
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras import models
# from tensorflow.keras import layers
# from tensorflow.keras.utils import to_categorical
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle


path = "Dataset/PetImages"

SIZE_list=[32, 64, 128,224]


import os

for SIZE in SIZE_list:

    images = []
    labels = []

        
    for folder in os.listdir("Dataset/PetImages"):
        for file_name in os.listdir(path+"/" + folder):
            if file_name.endswith(".jpg"):
                try:
                    image = cv2.imread(path+"/" + folder + "/" + file_name)
                    images.append(cv2.resize(image, (SIZE, SIZE)))
                    labels.append(folder)
                except:
                    print(file_name, folder)
                    pass
           
    data = {"images": images, "labels": labels}
    df = pd.DataFrame(data)
    
    with open("_dataset_" + str(SIZE)+".pickle", "wb") as f:
        pickle.dump(df, f)

    print("----------------------------------------")


# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()