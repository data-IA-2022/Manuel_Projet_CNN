import streamlit as st
import pandas as pd
from joblib import load
import index

index.load_image()
index.load_image_2()
# index.load_test_images()

st.title("Bienvenue sur la webapp ! 👋")

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
 