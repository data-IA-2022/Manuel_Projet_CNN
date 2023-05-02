import streamlit as st
import pandas as pd
from joblib import load
import urllib.request
import numpy as np

txt=""

@st.cache(suppress_st_warning=True)
def load_image():
    import cv2
    url = "https://pixnio.com/free-images/2017/03/23/2017-03-23-17-45-47.jpg"
    
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
def load_test_images(SIZE=160, SEED=31):
    # import cv2
    import pickle
    from sklearn.model_selection import train_test_split
    
    with open("dataset_" + str(SIZE)+".pickle", "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame.from_dict(data)

    train_df, test_df = train_test_split(df, 
                                         test_size=0.3,
                                         stratify=df['labels'],
                                         random_state=SEED)

    # le = LabelEncoder()

    # train_images = np.array(train_df['images'].to_list())#/ 255.0
    # train_labels = le.fit_transform(np.array(train_df['labels'].to_list()))

    test_images = np.array(test_df['images'].to_list())#/ 255.0
    test_labels = np.array(test_df['labels'].to_list())   

    return test_images, test_labels

def load_model(path):
    model = load(path)
    return model