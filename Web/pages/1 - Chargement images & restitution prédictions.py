import streamlit as st
import plotly.express as px
import Accueil as index
import cv2
import pandas as pd

#Titre de la form
st.title("Custom CNN !")

# Crée les onglets
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Training", 
                                        "Graf Model", 
                                        "Model Sumary", 
                                        "Model Code",
                                        "Predictions",
                                        'Score validation'])

# Onglet "Training"
with tab1:
    history_df = pd.DataFrame(index.history)
    fig = px.line(history_df, y=['accuracy', 'val_accuracy'], labels={'index':'Epoch', 'value':'Accuracy'}, 
          title='Training and Validation Accuracy')
    st.plotly_chart(fig, use_container_width=True)

# Onglet "Graf Model"
with tab2:
    col11, col12, col13 = st.columns([1,6,1])
   
    with col12:         
        st.image(st.session_state.model_graf_CNN_V2_4_0_1_91)

# Onglet "Model Summary"    
with tab3:
    col11, col12, col13 = st.columns([1,6,1])
   
    with col12:         
        st.session_state.model_CNN_V2_4_0_1_91.summary(print_fn=lambda x: st.text(x))

# Onglet "Model Code
with tab4:
    code = '''
            model = models.Sequential()
            model.add(layers.Conv2D(65, (2, 2), activation="relu", input_shape=df.iloc[0][0].shape))
            model.add(layers.MaxPool2D((2, 2)))
            model.add(layers.Conv2D(128, (2, 2), activation="relu"))
            model.add(layers.MaxPool2D((2, 2)))
            model.add(layers.Conv2D(256, (2, 2), activation="relu"))
            model.add(layers.MaxPool2D((2, 2)))
            model.add(layers.Conv2D(512, (2, 2), activation="sigmoid"))
            model.add(layers.MaxPool2D((2, 2)))
            model.add(layers.Conv2D(512, (2, 2), activation="relu"))
            model.add(layers.MaxPool2D((2, 2)))
            model.add(layers.Conv2D(512, (2, 2), activation="relu"))
            model.add(layers.MaxPool2D((3, 3)))

            model.add(layers.Flatten())

            model.add(layers.Dense(64, activation="relu"))
            model.add(layers.Dense(64, activation="relu"))
            model.add(layers.Dense(2, activation="softmax"))
            '''
    st.code(code, language='python')

# Onglet "Predictions"
with tab5:
        
    col11, col12 = st.columns([0.5,8])
    
    # Ajout d'un bouton radio pour sélectionner le type de prédiction
    with col12:     
        add_radio = st.radio(
            "Predictions :",
            index.lst,
            index = st.session_state.selector_type_image,
            horizontal = True)
         
    st.caption("")
    st.caption("")
    
    # Récupération de l'image en fonction de la prédiction sélectionnée
    if add_radio == index.lst[0]:   # index.load_image() # Si la prédiction sélectionnée est la première
        ld_image = index.img1       # Charger la première image
        st.session_state.selector_type_image=0 # Définir le type d'image sélectionné
  
    elif add_radio == index.lst[1]: # Sinon, si la prédiction sélectionnée est la deuxième
        ld_image = index.img2       # Charger la deuxième image
        st.session_state.selector_type_image=1 # Définir le type d'image sélectionné
    
    elif add_radio == index.lst[2]:  # Sinon, si la prédiction sélectionnée est la troisième
        hum_range = st.slider("Choisir une image :", 0, len(index.predict_reject_list)-1, 
                              st.session_state.index_bad_image) # Afficher un curseur pour sélectionner une image rejetée   
        ld_image=index.images[index.predict_reject_list[hum_range]] # Charger l'image sélectionnée
        ld_image = cv2.cvtColor(ld_image, cv2.COLOR_BGR2RGB) # Convertir l'image en RGB
        st.session_state.selector_type_image=2 # Définir le type d'image sélectionné
        st.session_state.index_bad_image = hum_range # Enregistrer l'index de l'image rejetée sélectionnée
        
    else:                           # Sinon, si la prédiction sélectionnée est la troisième
        hum_range = st.slider("Choisir une image :", 0, len(index.images)-1, 
                              st.session_state.index_image)  # Afficher un curseur pour sélectionner une image
        ld_image=index.images[hum_range] # Charger l'image sélectionnée
        ld_image = cv2.cvtColor(ld_image, cv2.COLOR_BGR2RGB) # Convertir l'image en RGB
        st.session_state.selector_type_image=3 # Définir le type d'image sélectionné
        st.session_state.index_image = hum_range # Enregistrer l'index de l'image sélectionnée
      
    col1, col2 = st.columns(2)
    
    # Redimensionnement de l'image à 160x160
    img_resized = cv2.resize(ld_image, (160, 160))    
    
    with col1:
        if st.session_state.selector_type_image==2:
            st.title("Image : #"+str(index.predict_reject_list[hum_range]))
            st.image(img_resized, caption='Image resizée 160x160')
        elif st.session_state.selector_type_image==3:
            st.title("Image : #"+str(hum_range))
            st.image(img_resized, caption='Image resizée 160x160')
        else:
            # Ajout de deux onglets pour afficher l'image redimensionnée et l'originale
            tab31, tab32 = st.tabs(["160x160", "Original"])
            with tab31:
                st.image(img_resized, caption='Image prise sur Internet')
            with tab32:
                st.image(ld_image, caption='Image prise sur Internet')
 
    with col2:
        st.title("")
        if st.session_state.selector_type_image==2:
            # Affichage de l'étiquette et de la prédiction pour l'image rejetée
            st.title("Label : "+index.lst_classes[index.labels[index.predict_reject_list[hum_range]]])
            st.title("Prédiction : "+index.lst_classes[ index.predict_list[index.predict_reject_list[hum_range]]])
        elif st.session_state.selector_type_image==3:
            st.title("Label : "+index.lst_classes[index.labels[hum_range]])
            st.title("Prédiction : "+index.lst_classes[ index.predict_list[hum_range]])
        else:
            st.title("")
            # Affichage de l'étiquette et de la prédiction pour l'image sélectionnée
            pred=index.prediction_2(img_resized, st.session_state.model_CNN_V2_4_0_1_91)
            st.session_state.predict_image=img_resized
            st.title("It's a "+index.lst_classes[pred]+" !")
            pass
        
# Onglet "Score des images de validation"
with tab6:
    st.caption("")
    st.caption("")
    st.image( cv2.imread('.\CNN_V2_4_0_1_91\Capture_valisation.png'))
    st.caption("")
    st.caption("")
    st.text('''
            Le score est probablement sur évalué, le memoire vidéo de la carte graphique
            étant visiblement insuffisante.''')