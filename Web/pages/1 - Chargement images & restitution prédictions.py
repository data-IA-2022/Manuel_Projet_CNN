import streamlit as st
import plotly.express as px
import Accueil as index
import cv2
import pandas as pd

st.title("Chargement de l'image !")

tab1, tab2, tab3, tab4 = st.tabs(["Training", "Graf CNN", "Sumary","Predictions"])

with tab1:
    history_df = pd.DataFrame(index.history)
    fig = px.line(history_df, y=['accuracy', 'val_accuracy'], labels={'index':'Epoch', 'value':'Accuracy'}, 
          title='Training and Validation Accuracy')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.image(st.session_state.model_graf, caption='Sunrise by the mountains')
    
with tab3:
    st.text(st.session_state.model.summary())

with tab4:
        
    add_radio = st.radio(
        "Choose image :",
        index.lst,
        index = st.session_state.selector_type_image,
        horizontal = True)
         
    if add_radio == index.lst[0]:
        ld_image = index.img1 # index.load_image()
        st.session_state.selector_type_image=0
    elif add_radio == index.lst[1]:
        ld_image = index.img2 # index.load_image_2()
        st.session_state.selector_type_image=1
    
    elif add_radio == index.lst[2]:
        hum_range = st.slider("Choose image :", 0, len(index.predict_reject_list)-1, 
                              st.session_state.index_bad_image)    
        ld_image=index.images[index.predict_reject_list[hum_range]]
        ld_image = cv2.cvtColor(ld_image, cv2.COLOR_BGR2RGB)
        st.session_state.selector_type_image=2
        st.session_state.index_bad_image = hum_range
        
    else:
        hum_range = st.slider("Choose image :", 0, len(index.images)-1, 
                              st.session_state.index_image)
        ld_image=index.images[hum_range]
        ld_image = cv2.cvtColor(ld_image, cv2.COLOR_BGR2RGB)
        st.session_state.selector_type_image=3
        st.session_state.index_image = hum_range
      
    col1, col2 = st.columns(2)
    
    img_resized = cv2.resize(ld_image, (160, 160))    
    
    with col1:
        if st.session_state.selector_type_image==2:
            st.title("Image : #"+str(index.predict_reject_list[hum_range]))
            st.image(img_resized, caption='Sunrise by the mountains')
        elif st.session_state.selector_type_image==3:
            st.title("Image : #"+str(hum_range))
            st.image(img_resized, caption='Sunrise by the mountains')
        else:
            tab31, tab32 = st.tabs(["160x160", "Original"])
            with tab31:
                st.image(img_resized, caption='Sunrise by the mountains')
            with tab32:
                st.image(ld_image, caption='Sunrise by the mountains')
 
    with col2:
        st.title("")
        if st.session_state.selector_type_image==2:
            st.title("Label : "+index.lst_classes[index.labels[index.predict_reject_list[hum_range]]])
            st.title("Prédiction : "+index.lst_classes[ index.predict_list[index.predict_reject_list[hum_range]]])
        elif st.session_state.selector_type_image==3:
            st.title("Label : "+index.lst_classes[index.labels[hum_range]])
            st.title("Prédiction : "+index.lst_classes[ index.predict_list[hum_range]])
        else:
            st.title("")
            pred=index.prediction_2(img_resized, st.session_state.model)
            st.session_state.predict_image=img_resized
            st.title("It's a "+index.lst_classes[pred]+" !")
            pass
     