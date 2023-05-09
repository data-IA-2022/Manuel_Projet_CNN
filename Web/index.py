import streamlit as st
# import pandas as pd
# from joblib import load
import urllib.request


txt=""

@st.cache(suppress_st_warning=True)
def load_image():
    import cv2
    import numpy as np
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
    import numpy as np
    url = "https://www.annuaire-animaux.net/images/fonds-ecran/maxi/chien-rigolo.jpg"
    
    img=[]
    with urllib.request.urlopen(url) as url_response:
        img_array = np.asarray(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    return img

@st.cache(suppress_st_warning=True)
def load_my_model_CNN_V2_4_0_1_91():
    from tensorflow.keras.models import load_model
    
    model = load_model('.\CNN_V2_4_0_1_91\model_.h5')
        
    return model
    
@st.cache(suppress_st_warning=True)
def load_model_graf_CNN_V2_4_0_1_91(model):
    from tensorflow.keras.utils import plot_model
    import cv2
    
    plot_model(model, to_file='.\CNN_V2_4_0_1_91\modele.png', show_shapes=True)    
    graf = cv2.imread('.\CNN_V2_4_0_1_91\modele.png')
   
    return graf

@st.cache(suppress_st_warning=True)
def prediction(num_image, images, model):
    import numpy as np
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
    import numpy as np
    pred = model.predict(np.array([image]))
    predicted_class = np.argmax(pred)
    print(pred)
    print(predicted_class)
    return predicted_class
    
@st.cache(suppress_st_warning=True)
def load_test_images(SIZE=160, SEED=31):
    import pickle
  
    with open("test_images" + str(SIZE)+".pickle", "rb") as f:
        test_images = pickle.load(f)
    
    with open("test_labels" + str(SIZE)+".pickle", "rb") as f:
        test_labels = pickle.load(f)
 
    return test_images, test_labels#, predict_list, predict_reject_list, lst_classes, history

@st.cache(suppress_st_warning=True)
def load_CNN_V2_4_0_1_91(SIZE=160, SEED=31):
    
    import pickle
    
    with open(".\CNN_V2_4_0_1_91\predict_list.pickle", "rb") as f:
        predict_list = pickle.load(f)
        
    with open(".\CNN_V2_4_0_1_91\predict_reject_list.pickle", "rb") as f:
        predict_reject_list = pickle.load(f)
        
    with open(".\CNN_V2_4_0_1_91\lst_classes.pickle", "rb") as f:
        lst_classes = pickle.load(f)

    with open(".\CNN_V2_4_0_1_91\history.pickle", "rb") as f:
        history = pickle.load(f)
    
    return predict_list, predict_reject_list, lst_classes, history
