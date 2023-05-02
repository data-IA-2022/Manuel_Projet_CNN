import streamlit as st
import pandas as pd
from joblib import load
import index

lst = ["Image 1", "Image 2", "Image 3"]

img1 = index.load_image()
img2 =index.load_image_2()
images, labels =index.load_test_images()

st.title("Bienvenue sur la webapp ! 👋")

if 'model' not in st.session_state:
    st.session_state['model'] = index.load_model()
    
if 'model_graf' not in st.session_state:
    st.session_state['model_graf'] = index.load_model_graf(st.session_state['model'])
    
if 'index_image' not in st.session_state:
    st.session_state['index_image'] = 10

if 'selector_type_image' not in st.session_state:
    st.session_state['selector_type_image'] = 2

# # Initialization
# if 'key' not in st.session_state:
#     st.title("Bienvenue ! 👋")
#     st.session_state['key'] = 'value'

# # Session State also supports attribute based syntax
# if 'key' in st.session_state:
#     st.title("Bienvenue sur la webapp ! 👋")
#     # st.session_state.key = 'value'

# @st.cache(suppress_st_warning=True)
# def load_data():
#     data =[1,2,3,4]
#     # data = pd.read_csv('Bike-Sharing-Dataset/day.csv')
#     # data = data[['dteday', 'holiday', 'weekday', 'weathersit', 'temp', 'hum', 'windspeed', 'cnt', 'season']]
#     # data['dteday'] = pd.to_datetime(data['dteday'], format='%Y-%m-%d')
#     # data['temp'] = data['temp'] * 41
#     # data['hum'] = data['hum'] * 100
#     # data['windspeed'] = data['windspeed'] * 67
#     # data['weekday'] = data['dteday'].dt.day_name()
#     # data['season'] = data['season'].map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'})
#     # data = data.join(pd.get_dummies(data['weekday'], prefix='weekday'))
#     return data


# def load_model(path):
#     model = load(path)
#     return model


# appel des données
# df = load_data()

# # affichage des données
# st.subheader("Aperçu des données")
# st.dataframe(df.head(100))
 