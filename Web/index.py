import streamlit as st
import pandas as pd
from joblib import load
import urllib.request
import numpy as np


txt=""

@st.cache(suppress_st_warning=True)
def load_image():
    import cv2
    url = "https://i.pinimg.com/originals/f7/71/47/f77147564a332c66f1759da52ac56ef5.jpg"
    
    img=[]
    with urllib.request.urlopen(url) as url_response:
        img_array = np.asarray(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    return img

@st.cache(suppress_st_warning=True)
def load_image_2():
    import cv2
    url = "https://www.annuaire-animaux.net/images/fonds-ecran/maxi/chien-rigolo.jpg"
    
    img=[]
    with urllib.request.urlopen(url) as url_response:
        img_array = np.asarray(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    return img

@st.cache(suppress_st_warning=True)
def load_my_model():
    from tensorflow.keras.models import load_model
    
    model = load_model('model_.h5')
        
    return model
    
@st.cache(suppress_st_warning=True)
def load_model_graf(model):
    from tensorflow.keras.utils import plot_model
    import cv2
    
    plot_model(model, to_file='modele.png', show_shapes=True)    
    graf = cv2.imread('modele.png')
   
    return graf

@st.cache(suppress_st_warning=True)
def prediction(num_image, images, model):
    from tensorflow.keras import backend
    from tensorflow.keras.models import load_model
    print(images[num_image].shape)
    # model = load_model('model_.h5')
    model.summary()
    pred = model.predict(np.array([images[num_image]]))
    predicted_class = np.argmax(pred)
    print(pred)
    print(predicted_class)
    # backend.clear_session()
    return 1

@st.cache(suppress_st_warning=True)
def prediction_2(image, model):
    from tensorflow.keras import backend
    from tensorflow.keras.models import load_model
    
    pred = model.predict(np.array([image]))
    predicted_class = np.argmax(pred)
    print(pred)
    print(predicted_class)
    return predicted_class
    
@st.cache(suppress_st_warning=True)
def load_test_images(path, SIZE=160, SEED=31):
    # import cv2
    import pickle
    # from sklearn.model_selection import train_test_split
    print(path)
    
    with open("test_images" + str(SIZE)+".pickle", "rb") as f:
        test_images = pickle.load(f)
    
    with open("test_labels" + str(SIZE)+".pickle", "rb") as f:
        test_labels = pickle.load(f)
    
    with open("predict_list.pickle", "rb") as f:
        predict_list = pickle.load(f)
        
    with open("predict_reject_list.pickle", "rb") as f:
        predict_reject_list = pickle.load(f)
        
    with open("lst_classes.pickle", "rb") as f:
        lst_classes = pickle.load(f)

    with open("history.pickle", "rb") as f:
        history = pickle.load(f)
    
    return test_images, test_labels, predict_list, predict_reject_list, lst_classes, history
