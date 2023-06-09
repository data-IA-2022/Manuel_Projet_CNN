import streamlit as st

# Fonction pour charger un modèle sauvegardé
@st.cache(suppress_st_warning=True)
def load_my_model_CNN_V2_4_0_1_91():
    from tensorflow.keras.models import load_model
    
    model = load_model('.\CNN_V2_4_0_1_91\model_.h5')
        
    return model
    
# Fonction pour charger le modèle sauvegardé et le représenter sous forme de graph
@st.cache(suppress_st_warning=True)
def load_model_graf_CNN_V2_4_0_1_91(model):
    from tensorflow.keras.utils import plot_model
    import cv2
    
    plot_model(model, to_file='.\CNN_V2_4_0_1_91\modele.png', show_shapes=True)    
    graf = cv2.imread('.\CNN_V2_4_0_1_91\modele.png')
   
    return graf

# Fonction pour charger les images de test sauvegardées et leurs labels  
@st.cache(suppress_st_warning=True)
def load_test_images(SIZE=160, SEED=31):
    import pickle
  
    with open("./CNN_V2_4_0_1_91/test_images" + str(SIZE)+".pickle", "rb") as f:
        test_images = pickle.load(f)

    return test_images

# Fonction pour effectuer une prédiction sur une image donnée avec un modèle donné
@st.cache(suppress_st_warning=True)
def prediction_2(image, model):
    import numpy as np
    pred = model.predict(np.array([image]))
    predicted_class = np.argmax(pred)
    print(pred)
    print(predicted_class)
    return pred
 
# Fonction pour charger les resultats précalculés 
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
