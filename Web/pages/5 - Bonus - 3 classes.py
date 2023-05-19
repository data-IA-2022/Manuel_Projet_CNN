import streamlit as st
import plotly.express as px
import pandas as pd
import pickle
import plotly.graph_objs as go

def plot_train_graph(path):
    
    history=None
    with open(path + "history.pickle", "rb") as f:
        history = pickle.load(f)
    
    history_df = pd.DataFrame(history)
    fig = px.line(history_df, y=['accuracy', 'val_accuracy'], labels={'index':'Epoch', 'value':'Accuracy'}, 
          title='Training and Validation Accuracy')
    fig.update_yaxes(range=[0.5, 1])
    
    return fig  

    
def plot_matrice_confus_3c(path):
    
    confusion_matrix=None
    
    # Chargement de la matrice de confusion à partir du fichier
    with open(path + "confusion_matrix.pickle", 'rb') as f:
        confusion_matrix = pickle.load(f)
    df = pd.DataFrame(confusion_matrix, columns=['Cat','Dog','Horse'], index=['Cat','Dog','Horse'])
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
st.markdown("<h1 style='text-align: center; color: grey;'>Bonus - Custom CNN 3 classes</h1>", unsafe_allow_html=True)

# # Création de 2 onglets
# tab1, tab2 = st.tabs(["Resnet50", 
#                       "ModilNetV2"])

with_data = not st.sidebar.checkbox('Avec data augmentation')

if with_data:
    st.title("Trainning.")
   
    st.plotly_chart(plot_train_graph(".\CNN_V2_4_0_1_91_3_class\\"), use_container_width=True)
   
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
        st.plotly_chart(plot_matrice_confus_3c(".\CNN_V2_4_0_1_91_3_class\\"), use_container_width=True)
      
    with col12:     
            
        st.header("Métriques.")
        
        with open('.\CNN_V2_4_0_1_91_3_class\performance.pickle', 'rb') as f:
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
        
else:
    st.title("Trainning.")
   
    st.plotly_chart(plot_train_graph(".\CNN_V2_4_0_1_91_3_class_aug_horse\\"), use_container_width=True)
   
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
        st.plotly_chart(plot_matrice_confus_3c(".\CNN_V2_4_0_1_91_3_class_aug_horse\\"), use_container_width=True)
   
                
    with col12:     
            
        st.header("Métriques.")
        
        with open('.\CNN_V2_4_0_1_91_3_class_aug_horse\performance.pickle', 'rb') as f:
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

