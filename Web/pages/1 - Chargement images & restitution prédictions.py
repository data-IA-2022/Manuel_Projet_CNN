import datetime
import streamlit as st
import plotly.express as px
# import numpy as np
import Accueil as index
import cv2

st.title("Chargement de l'image !")

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose image :",
        index.lst,
        index = st.session_state.selector_type_image
    )
     
if add_radio == index.lst[0]:
    ld_image = index.img1 # index.load_image()
    st.session_state.selector_type_image=0
elif add_radio == index.lst[1]:
    ld_image = index.img2 # index.load_image_2()
    st.session_state.selector_type_image=1
else:
    hum_range = st.slider("Choose image :", 0, len(index.images), st.session_state.index_image)
    st.title(hum_range)
    ld_image=index.images[hum_range]
    ld_image = cv2.cvtColor(ld_image, cv2.COLOR_BGR2RGB)
    st.session_state.selector_type_image=2
    st.session_state.index_image = hum_range
 
if ld_image.shape != (160, 160, 3):
    col1, col2,col3 = st.columns(3)
else:
    col1, col2 = st.columns(2)

img_resized = cv2.resize(ld_image, (160, 160))    

with col1:
    st.image(img_resized, caption='Sunrise by the mountains')

if ld_image.shape != (160, 160, 3):
    with col2:
        st.image(ld_image, caption='Sunrise by the mountains')
        st.image(ld_image, caption='Sunrise by the mountains')
    with col3:
        st.image(st.session_state.model_graf, caption='Sunrise by the mountains')
else:
    with col2:
        st.image(st.session_state.model_graf, caption='Sunrise by the mountains')




# # appel des données
# df = load_data()

# # champs de filtrage
# temp_range = st.sidebar.slider('Valeurs de température', min(df['temp']), max(df['temp']),
#                                [min(df['temp']), max(df['temp'])])
# hum_range = st.sidebar.slider("Valeurs d'humidité", min(df['hum']), max(df['hum']), [min(df['hum']), max(df['hum'])])
# windspeed_range = st.sidebar.slider('Valeurs de vitesse du vent', min(df['windspeed']), max(df['windspeed']),
#                                     [min(df['windspeed']), max(df['windspeed'])])
# weekday_values = st.sidebar.multiselect('Jours de la semaine', df['weekday'].unique(), df['weekday'].unique())
# date_values = st.sidebar.date_input('Dates', [datetime.date(2011, 1, 1), datetime.date(2011, 12, 31)],
#                                     datetime.date(2011, 1, 1), datetime.date(2011, 12, 31))

# # si aucune date de fin n'est renseignée, on assigne le 31/12/2011
# if len(date_values) == 1:
#     date_values = (date_values[0], datetime.date(2011, 12, 31))

# # filtrage des données
# df_filter = df[
#     (df['dteday'].between(date_values[0].strftime('%Y-%m-%d'), date_values[1].strftime('%Y-%m-%d'))) &
#     (df['temp'].between(temp_range[0], temp_range[1])) &
#     (df['hum'].between(hum_range[0], hum_range[1])) &
#     (df['windspeed'].between(windspeed_range[0], windspeed_range[1])) &
#     (df['weekday'].isin(weekday_values))
#     ]

# # nuage de points
# st.subheader("Evolution du nombre d'utilisations quotidiennes")
# fig = px.scatter(df_filter, x="dteday", y="cnt", color="temp", color_continuous_scale="thermal")
# st.plotly_chart(fig)

# # diagramme en barres
# st.subheader("Moyenne du nombre d'utilisations selon le jour de la semaine")
# fig = px.histogram(df_filter, x='weekday', y='cnt', histfunc='avg')
# st.plotly_chart(fig)

# # nuage de points
# st.subheader("Lien entre la vitesse du vent et le nombre d'utilisations")
# fig = px.scatter(df_filter, x="windspeed", y="cnt", trendline="ols")
# st.plotly_chart(fig)

# # métrique
# st.metric(label="Corrélation entre la vitesse du vent et le nombre d'utilisations",
#           value=np.round(df_filter['cnt'].corr(df_filter['windspeed']), 2))

# # violin plot
# st.subheader("Distribution du nombre d'utilisations selon la saison")
# fig = px.violin(df_filter, y="cnt", x="season", color="season", box=True,
#                 color_discrete_sequence=["#87CEEB", "#90EE90", "#FFA07A", "#FFDAB9"])
# st.plotly_chart(fig)
