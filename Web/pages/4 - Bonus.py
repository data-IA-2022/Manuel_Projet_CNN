import streamlit as st
import plotly.express as px
import pandas as pd
import pickle
import Accueil as index
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
st.markdown("<h1 style='text-align: center; color: grey;'>Bonus</h1>", unsafe_allow_html=True)

# Création de 2 onglets
tab1, tab2 = st.tabs(["Resnet50", 
                      "ModilNetV2"])

lst=["Trainning", "Fine tunning", "Transfère learning"]


add_radio = st.sidebar.radio(
    "Mode d'apprentissage : ",
    lst,
    index = 0,
    horizontal = False)

with tab1:
    
   if add_radio == lst[0]:
       
       st.title("Trainning.")
       
       st.plotly_chart(plot_train_graph(".\Resnet50\CNN Resnet50 train\\"), use_container_width=True)
       
       st.title("Model.")
       code = '''
        model = ResNet50 (include_top=True,
                    weights=None,
                    input_tensor=None,
                    input_shape=(SIZE,SIZE,3),
                    pooling='max',
                    classes=np.max(train_labels+1))
               '''
       st.code(code, language='python')
       
       st.title("Performance.")
       
       col11, col12 = st.columns(2)
       
       with col11:     
       
           st.header("Matrice de confusion.")
           st.plotly_chart(plot_matrice_confus(".\Resnet50\CNN Resnet50 train\\"), use_container_width=True)
           
       with col12:     
                
            st.header("Métriques.")
            
            with open('.\Resnet50\CNN Resnet50 train\performance.pickle', 'rb') as f:
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
       
       
   elif add_radio == lst[1]: 
       
       st.title("Fine tunning.")
       
       st.plotly_chart(plot_train_graph(".\Resnet50\CNN Resnet50 fine\\"), use_container_width=True)
       
       st.title("Model.")
       code = '''
        model = ResNet50 (include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=(SIZE,SIZE,3),
                    pooling='max',
                    classes=np.max(train_labels+1))
               '''
       st.code(code, language='python')
       
       st.title("Performance.")
       
       col11, col12 = st.columns(2)
       
       with col11:     
       
           st.header("Matrice de confusion.")
           st.plotly_chart(plot_matrice_confus(".\Resnet50\CNN Resnet50 fine\\"), use_container_width=True)
           
       with col12:     
                
            st.header("Métriques.")
            
            with open('.\Resnet50\CNN Resnet50 fine\performance.pickle', 'rb') as f:
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
       
       st.title("Transfere learning.")
       
       st.plotly_chart(plot_train_graph(".\Resnet50\CNN Trans Learn 244x244 batch_5\\"), use_container_width=True)
       
       st.title("Model.")
       code = '''
        model = Sequential()

        model.add(ResNet50 (include_top=False,
                          weights="imagenet",
                          pooling='max'))

        # model.add(Flatten())

        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation = 'softmax'))
        model.layers[0].trainable = False
               '''
       st.code(code, language='python')
       
       col11, col12 = st.columns(2)
       
       with col11:     
       
           st.header("Matrice de confusion.")
           st.plotly_chart(plot_matrice_confus(".\Resnet50\CNN Trans Learn 244x244 batch_5\\"), use_container_width=True)
           
       with col12:     
                
            st.header("Métriques.")
            
            with open('.\Resnet50\CNN Resnet50 fine\CNN Trans Learn 244x244 batch_5', 'rb') as f:
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
      
    if add_radio == lst[0]:
        # Création de 2 onglets
        tab11, tab12, tab13 = st.tabs(["Alpha = 1", 
                              "Alpha = 0.35",
                              "Alpha = 1 (64x64)"])
        
        with tab11: 
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Training\MobileNetV2\\"), use_container_width=True)
            
        with tab12:
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Training\MobileNetV2 0.35\\"), use_container_width=True)
        
        with tab13:
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Training\MobileNetV2 64x64 - 5\\"), use_container_width=True)
               
    elif add_radio == lst[1]: 
        # Création de 2 onglets
        tab11, tab12 = st.tabs(["Alpha = 1", 
                              "Alpha = 0.35"])
        
        with tab11:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Fine tunning\MobileNetV2 Imagenet\\"), use_container_width=True)
            
        with tab12:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Fine tunning\MobileNetV2 Imagenet 0.35\\"), use_container_width=True)
            
    else:   
        # Création de 4 onglets
        tab11, tab12, tab13, tab14 = st.tabs(["Alpha = 1 (160x160)", 
                              "Alpha = 0.35 (160x160)",
                              "Alpha = 1 (224x224)", 
                              "Alpha = 0.35 (224x224)"])
        with tab11:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Transfere learning\MobileNetV2 160x160\\"), use_container_width=True)
            
        with tab12:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Transfere learning\MobileNetV2 160x160 - 0.35\\"), use_container_width=True)
            
        with tab13:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Transfere learning\MobileNetV2 224x224\\"), use_container_width=True)
            
        with tab14:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Transfere learning\MobileNetV2 224x224 - 0.35\\"), use_container_width=True)
            
# # Titre de la form
# st.title("Mode d'apprentissage des réseaux de neurones convolutifs (CNN).")

# st.caption("")
# st.caption("")

# col11, col12, col13 = st.columns([1,6,1])
 
# # Affichage d'une image de réseau de neurone propagation retro-propagation  
# with col12:        
#     st.markdown("![Alt Text](https://i.makeagif.com/media/7-23-2019/q3ItDm.gif)")
    
# st.caption("")
# st.caption("")

# # Description sur les CNN
# st.text('''
#         Les réseaux de neurones convolutifs (CNN) sont des systèmes d'intelligence 
#         artificielle qui pprennent à reconnaître des motifs et des formes dans des images.
        
#         Le processus d'apprentissage des CNN se fait en deux étapes. Tout d'abord, le 
#         réseau est entraîné sur un ensemble de données d'images pour apprendre à 
#         reconnaître les motifs et les caractéristiques des différentes classes d'images
#         (par exemple, chats, chiens, oiseaux).

#         Pendant cette première étape, le réseau va ajuster les poids de chaque neurone
#         de chaque couche pour minimiser l'erreur de classification entre la sortie prédite
#         par le réseau et la classe réelle de l'image. Cette étape peut prendre beaucoup de
#         temps et nécessite souvent de grandes quantités de données.''')

# col11, col12, col13 = st.columns([0.1,6,0.1])
   
# # Affichage d'une autre image de réseau de neurone propagation retro-propagation 
# with col12:        
#     st.markdown("![Alt Text](https://miro.medium.com/v2/resize:fit:640/1*mTTmfdMcFlPtyu8__vRHOQ.gif)")
 
# # Description sur les CNN
# st.text('''
#         Une fois que le réseau a été entraîné, il peut être utilisé pour classifier de 
#         nouvelles images. Lorsqu'une nouvelle image est présentée au réseau, celui-ci
#         utilise les caractéristiques qu'il a apprises lors de l'étape d'entraînement pour 
#         prédire la classe de l'image.''')

# st.caption("")
# st.caption("")

# col11, col12, col13 = st.columns([1,6,1])
   
# # Affichage d'une image de réseau de neurone propagation
# with col12:        
#     st.markdown("![Alt Text](https://miro.medium.com/v2/resize:fit:500/0*61ZaNNpbpMtZLLpZ.)")

# st.caption("")
# st.caption("")

# # Conclusion sur les CNN
# st.text('''
#         En résumé, le processus d'apprentissage des CNN se fait en deux étapes : 
#         l'entraînement sur un ensemble de données d'images pour apprendre à reconnaître 
#         les motifs et les caractéristiques des différentes classes d'images, et 
#         l'utilisation du réseau entraîné pour classifier de nouvelles images en utilisant 
#         les caractéristiques qu'il a apprises. Ce processus permet aux CNN de reconnaître 
#         des objets dans des images avec une grande précision.''')
