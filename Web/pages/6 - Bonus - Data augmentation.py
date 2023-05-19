import streamlit as st
import plotly.express as px
import pandas as pd
import pickle
import Accueil as index
import plotly.graph_objs as go

def plot_train_graph(path, mini=0.5):
    
    history=None
    with open(path + "history.pickle", "rb") as f:
        history = pickle.load(f)
    
    history_df = pd.DataFrame(history)
    fig = px.line(history_df, y=['accuracy', 'val_accuracy'], labels={'index':'Epoch', 'value':'Accuracy'}, 
          title='Training and Validation Accuracy')
    fig.update_yaxes(range=[mini, 1])
    
    return fig  

def plot_matrice_confus(path):
    
    confusion_matrix=None
    
    # Chargement de la matrice de confusion à partir du fichier
    with open(path + "confusion_matrix.pickle", 'rb') as f:
        confusion_matrix = pickle.load(f)
    df = pd.DataFrame(confusion_matrix, columns=index.lst_classes, index=index.lst_classes[::-1])
    print (df)
    
    # Création d'une trace Plotly pour la matrice de confusion
    trace = go.Heatmap(z=df.values[::-1],
                       x=df.columns,
                       y=df.index,
                       colorscale='amp')
    
    # Création d'une figure Plotly et ajout de la trace
    fig = go.Figure(data=[trace])
    
    return fig 
    
# Affichage du titre
st.markdown("<h1 style='text-align: center; color: grey;'>Bonus - Data augmentation</h1>", unsafe_allow_html=True)

# Création de 2 onglets
tab1, tab2, tab3 = st.tabs(["Custom CNN",
                      "Resnet50", 
                      "ModilNetV2"])

with tab1:
    st.title("Trainning.")
   
    st.plotly_chart(plot_train_graph("CNN_V2_4_0_1_91_2 class_aug\\"), use_container_width=True)
   
    st.title("Model.")
    code = '''
    model = ResNet50 (include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=(SIZE,SIZE,3),
                pooling='max',
                classes=np.max(train_labels+1))
    
    Total params: 23,587,712
    Trainable params: 23,534,592
    Non-trainable params: 53,120
            '''
    st.code(code, language='python')
   
    st.title("Performance.")
   
    col11, col12 = st.columns(2)
   
    with col11:     
   
        st.header("Matrice de confusion.")
        st.plotly_chart(plot_matrice_confus(".\CNN_V2_4_0_1_91_2 class_aug\\"), use_container_width=True)
      
    with col12:     
            
        st.header("Métriques.")
        
        with open('.\CNN_V2_4_0_1_91_2 class_aug\performance.pickle', 'rb') as f:
            performance=pickle.load(f)
           
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        
        st.subheader("Precision : " + str(round(performance[0]*100, 3)) +" %")
        st.subheader("Recall : " + str(round(performance[1]*100, 3)) +" %")
        st.subheader("f_score : " + str(round(performance[2]*100, 3)) +" %")

with tab2:
    
    st.title("Transfer learning.")
   
    st.plotly_chart(plot_train_graph(".\Resnet50\CNN Trans Learn 160x160 aug\\"), use_container_width=True)
   
    st.title("Model.")
    code = '''
    model = ResNet50 (include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=(SIZE,SIZE,3),
                pooling='max',
                classes=np.max(train_labels+1))
    
    Total params: 23,587,712
    Trainable params: 23,534,592
    Non-trainable params: 53,120
            '''
    st.code(code, language='python')
   
    st.title("Performance.")
   
    col11, col12 = st.columns(2)
   
    with col11:     
   
        st.header("Matrice de confusion.")
        st.plotly_chart(plot_matrice_confus(".\Resnet50\CNN Trans Learn 160x160 aug\\"), use_container_width=True)
      
    with col12:     
            
        st.header("Métriques.")
        
        with open('.\Resnet50\CNN Trans Learn 160x160 aug\performance.pickle', 'rb') as f:
            performance=pickle.load(f)
           
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        
        st.subheader("Precision : " + str(round(performance[0]*100, 3)) +" %")
        st.subheader("Recall : " + str(round(performance[1]*100, 3)) +" %")
        st.subheader("f_score : " + str(round(performance[2]*100, 3)) +" %")
        
with tab3:
    tab11, tab21 = st.tabs(["Alpha=1", "Alpha=0.35"])
    
    with tab11:
        
        st.title("Fine tuning alpha=1.")
       
        st.plotly_chart(plot_train_graph(".\MobileNetV2\Fine tunning\MobileNetV2 Imagenet class aug\\"), use_container_width=True)
       
        st.title("Model.")
        code = '''
        model = ResNet50 (include_top=True,
                    weights=None,
                    input_tensor=None,
                    input_shape=(SIZE,SIZE,3),
                    pooling='max',
                    classes=np.max(train_labels+1))
        
        Total params: 23,587,712
        Trainable params: 23,534,592
        Non-trainable params: 53,120
                '''
        st.code(code, language='python')
       
        st.title("Performance.")
       
        col11, col12 = st.columns(2)
       
        with col11:     
       
            st.header("Matrice de confusion.")
            st.plotly_chart(plot_matrice_confus(".\MobileNetV2\Fine tunning\MobileNetV2 Imagenet class aug\\"), use_container_width=True)
          
        with col12:     
                
            st.header("Métriques.")
            
            with open('.\MobileNetV2\Fine tunning\MobileNetV2 Imagenet class aug\performance.pickle', 'rb') as f:
                performance=pickle.load(f)
               
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            
            st.subheader("Precision : " + str(round(performance[0]*100, 3)) +" %")
            st.subheader("Recall : " + str(round(performance[1]*100, 3)) +" %")
            st.subheader("f_score : " + str(round(performance[2]*100, 3)) +" %")
   
    with tab21:
        
        st.title("Fine tuning alpha=0.3")
       
        st.plotly_chart(plot_train_graph(".\MobileNetV2\Fine tunning\MobileNetV2Imagenet 0.35 aug\\"), use_container_width=True)
       
        st.title("Model.")
        code = '''
        model = ResNet50 (include_top=True,
                    weights=None,
                    input_tensor=None,
                    input_shape=(SIZE,SIZE,3),
                    pooling='max',
                    classes=np.max(train_labels+1))
        
        Total params: 23,587,712
        Trainable params: 23,534,592
        Non-trainable params: 53,120
                '''
        st.code(code, language='python')
       
        st.title("Performance.")
       
        col11, col12 = st.columns(2)
       
        with col11:     
       
            st.header("Matrice de confusion.")
            st.plotly_chart(plot_matrice_confus(".\MobileNetV2\Fine tunning\MobileNetV2Imagenet 0.35 aug\\"), use_container_width=True)
          
        with col12:     
                
            st.header("Métriques.")
            
            with open('.\MobileNetV2\Fine tunning\MobileNetV2Imagenet 0.35 aug\performance.pickle', 'rb') as f:
                performance=pickle.load(f)
               
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            
            st.subheader("Precision : " + str(round(performance[0]*100, 3)) +" %")
            st.subheader("Recall : " + str(round(performance[1]*100, 3)) +" %")
            st.subheader("f_score : " + str(round(performance[2]*100, 3)) +" %")
               