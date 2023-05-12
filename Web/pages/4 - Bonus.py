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
        
        Total params: 23,587,712
        Trainable params: 23,534,592
        Non-trainable params: 53,120
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
        
        Total params: 23,591,810
        Trainable params: 23,538,690
        Non-trainable params: 53,120
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
       
        
       Total params: 23,723,138
       Trainable params: 135,426
       Non-trainable params: 23,587,712
               '''
       st.code(code, language='python')
       
       col11, col12 = st.columns(2)
       
       with col11:     
       
           st.header("Matrice de confusion.")
           st.plotly_chart(plot_matrice_confus(".\Resnet50\CNN Trans Learn 244x244 batch_5\\"), use_container_width=True)
           
       with col12:     
                
            st.header("Métriques.")
            
            with open('.\Resnet50\CNN Trans Learn 244x244 batch_5\performance.pickle', 'rb') as f:
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
                              "Alpha = 5 (64x64)"])
        
        with tab11: 
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Training\MobileNetV2\\"), use_container_width=True)
            
            st.title("Model.")
            code = '''
            model=MobileNetV2(include_top=True,
                                weights=None,
                                input_tensor=None,
                                input_shape=(SIZE,SIZE,3),
                                pooling='max',
                                classes=np.max(train_labels+1))
             
              
            Total params: 2,260,546
            Trainable params: 2,226,434
            Non-trainable params: 34,112
                    '''
            st.code(code, language='python')
            
            col11, col12 = st.columns(2)
            
            with col11:     
            
                st.header("Matrice de confusion.")
                st.plotly_chart(plot_matrice_confus(".\MobileNetV2\Training\MobileNetV2\\"), use_container_width=True)
                
            with col12:     
                     
                  st.header("Métriques.")
                 
                  with open('.\MobileNetV2\Training\MobileNetV2\performance.pickle', 'rb') as f:
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
            
        with tab12:
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Training\MobileNetV2 0.35\\"), use_container_width=True)
        
            st.title("Model.")
            code = '''
            model=MobileNetV2(include_top=True,
                              weights=None,
                              input_tensor=None,
                              alpha=0.35,
                              input_shape=(SIZE,SIZE,3),
                              pooling='max',
                              classes=np.max(train_labels+1))
             
              
            Total params: 412,770
            Trainable params: 398,690
            Non-trainable params: 14,080
                    '''
            st.code(code, language='python')
            
            col11, col12 = st.columns(2)
            
            with col11:     
            
                st.header("Matrice de confusion.")
                st.plotly_chart(plot_matrice_confus(".\MobileNetV2\Training\MobileNetV2 0.35\\"), use_container_width=True)
                
            with col12:     
                     
                  st.header("Métriques.")
                 
                  with open('.\MobileNetV2\Training\MobileNetV2 0.35\performance.pickle', 'rb') as f:
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
        with tab13:
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Training\MobileNetV2 64x64 - 5\\"), use_container_width=True)
               
        
            st.title("Model.")
            code = '''
            model=MobileNetV2(include_top=True,
                              weights=None,
                              input_tensor=None,
                              alpha=5,
                              input_shape=(SIZE,SIZE,3),
                              pooling='max',
                              classes=np.max(train_labels+1))
             
              
            Total params: 53,796,162
            Trainable params: 53,625,602
            Non-trainable params: 170,560
                    '''
            st.code(code, language='python')
            
            col11, col12 = st.columns(2)
            
            with col11:     
            
                st.header("Matrice de confusion.")
                st.plotly_chart(plot_matrice_confus(".\MobileNetV2\Training\MobileNetV2 64x64 - 5\\"), use_container_width=True)
                
                
            with col12:     
                     
                  st.header("Métriques.")
                 
                  with open('.\MobileNetV2\Training\MobileNetV2 64x64 - 5\performance.pickle', 'rb') as f:
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
        # Création de 2 onglets
        tab11, tab12 = st.tabs(["Alpha = 1", 
                              "Alpha = 0.35"])
        
        with tab11:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Fine tunning\MobileNetV2 Imagenet\\"), use_container_width=True)
            
            st.title("Model.")
            code = '''
            model=MobileNetV2(include_top=False,
                              weights='imagenet',
                              input_tensor=None,
                              input_shape=(SIZE,SIZE,3),
                              pooling='max',
                              classes=np.max(train_labels+1))
        
            Total params: 2,257,984
            Trainable params: 2,223,872
            Non-trainable params: 34,112
                    '''
            st.code(code, language='python')
            
            col11, col12 = st.columns(2)
            
            with col11:     
            
                st.header("Matrice de confusion.")
                st.plotly_chart(plot_matrice_confus(".\MobileNetV2\Fine tunning\MobileNetV2 Imagenet\\"), use_container_width=True)
                
            with col12:     
                     
                  st.header("Métriques.")
                 
                  with open('.\MobileNetV2\Fine tunning\MobileNetV2 Imagenet\performance.pickle', 'rb') as f:
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
            
        with tab12:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Fine tunning\MobileNetV2 Imagenet 0.35\\"), use_container_width=True)
            
            st.title("Model.")
            code = '''
            model=MobileNetV2(include_top=False,
                              weights='imagenet',
                              alpha=0.35,
                              input_tensor=None,
                              input_shape=(SIZE,SIZE,3),
                              pooling='max',
                              classes=np.max(train_labels+1))
        
            Total params: 410,208
            Trainable params: 396,128
            Non-trainable params: 14,080
                    '''
            st.code(code, language='python')
            
            col11, col12 = st.columns(2)
            
            with col11:     
            
                st.header("Matrice de confusion.")
                st.plotly_chart(plot_matrice_confus(".\MobileNetV2\Fine tunning\MobileNetV2 Imagenet 0.35\\"), use_container_width=True)
                
            with col12:     
                     
                  st.header("Métriques.")
                 
                  with open('.\MobileNetV2\Fine tunning\MobileNetV2 Imagenet 0.35\performance.pickle', 'rb') as f:
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
        # Création de 4 onglets
        tab11, tab12, tab13, tab14 = st.tabs(["Alpha = 1 (160x160)", 
                              "Alpha = 0.35 (160x160)",
                              "Alpha = 1 (224x224)", 
                              "Alpha = 0.35 (224x224)"])
        with tab11:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Transfere learning\MobileNetV2 160x160\\"), use_container_width=True)
            
            st.title("Model.")
            code = '''
            model = Sequential()

            model.add(MobileNetV2 (include_top=False,
                              weights="imagenet",
                              alpha=1.0,
                              pooling='max'))

            # model.add(Flatten())

            model.add(Dense(64, activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(2, activation = 'softmax'))
            model.layers[0].trainable = False
        
        
            Total params: 2,344,258
            Trainable params: 86,274
            Non-trainable params: 2,257,984
                    '''
            st.code(code, language='python')
            
            col11, col12 = st.columns(2)
            
            with col11:     
            
                st.header("Matrice de confusion.")
                st.plotly_chart(plot_matrice_confus(".\MobileNetV2\Transfere learning\MobileNetV2 160x160\\"), use_container_width=True)
                
            with col12:     
                     
                  st.header("Métriques.")
                 
                  with open('.\MobileNetV2\Transfere learning\MobileNetV2 160x160\performance.pickle', 'rb') as f:
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
            
        with tab12:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Transfere learning\MobileNetV2 160x160 - 0.35\\"), use_container_width=True)
            
        with tab13:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Transfere learning\MobileNetV2 224x224\\"), use_container_width=True)
            
        with tab14:
            
            st.plotly_chart(plot_train_graph(".\MobileNetV2\Transfere learning\MobileNetV2 224x224 - 0.35\\"), use_container_width=True)
            