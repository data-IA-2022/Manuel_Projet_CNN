import streamlit as st
import pandas as pd
from joblib import load
import index
# from tensorflow.keras.models import load_model

lst = ["Image 1", "Image 2", "Images de test mal classées", "Toutes les images de test"]

img1 = index.load_image()
img2 =index.load_image_2()
images, labels, predict_list, predict_reject_list, lst_classes, history =index.load_test_images('\CNN V2.4 0.1 91%')
img=None



if 'model' not in st.session_state:
    st.session_state['model'] = index.load_my_model()
    
if 'model_graf' not in st.session_state:
    st.session_state['model_graf'] = index.load_model_graf(st.session_state['model'])
    
if 'index_image' not in st.session_state:
    st.session_state['index_image'] = 10
    
if 'index_bad_image' not in st.session_state:
    st.session_state['index_bad_image'] = 10

if 'selector_type_image' not in st.session_state:
    st.session_state['selector_type_image'] = 2

if 'predict_image' not in st.session_state:    
    st.session_state['img']=img
     
def prediction(img, images, model):
    return index.prediction(img, images, model)
    
def prediction_2(img, model):
    return index.prediction_2(img, model)
    
st.title("Bienvenue sur la webapp ! 👋")
