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
        
        
    st.title("Model.")
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
   model.add(layers.Conv2D(880, (2, 2), activation="relu"))
   model.add(layers.MaxPool2D((2, 2)))
   model.add(layers.Conv2D(880, (2, 2), activation="relu"))
   model.add(layers.MaxPool2D((3, 3)))

   model.add(layers.Flatten())

   model.add(layers.Dense(64, activation="relu"))
   model.add(layers.Dense(64, activation="relu"))
   model.add(layers.Dense(3, activation="softmax"))

   model.summary()

   print("Model")

   optimizer="adamax"

   # model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

   datagen = ImageDataGenerator(#rescale=1/255.,
                                rotation_range=20,
                                zoom_range=0.15,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.15,
                                horizontal_flip=True,
                                validation_split=0.2)

   training_generator = datagen.flow(xx_train, 
                                     yy_train, 
                                     batch_size=BATCH_SIZE,
                                     subset='training',
                                     seed=5)

   validation_generator = datagen.flow(xx_train, 
                                       yy_train, 
                                       batch_size=BATCH_SIZE,
                                       subset='validation',
                                       seed=5)

   history = model.fit_generator(training_generator,
                                 steps_per_epoch=(len(xx_train)*0.8)//BATCH_SIZE, 
                                 epochs=25, 
                                 validation_data=validation_generator, 
                                 validation_steps=(len(xx_train)*0.2)//BATCH_SIZE)
            '''
    st.code(code, language='python')
   
    
else:
    st.title("Trainning.")
   
    st.plotly_chart(plot_train_graph(".\CNN_V2_4_0_1_91_3_class_aug_horse\\"), use_container_width=True)
   
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
        
    st.title("Model.")
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
    model.add(layers.Conv2D(880, (2, 2), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(880, (2, 2), activation="relu"))
    model.add(layers.MaxPool2D((3, 3)))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))

    model.summary()

    model.compile(optimizer="adamax",#adamax,
                  loss="SparseCategoricalCrossentropy",
                  metrics=["accuracy"])

    history = model.fit(train_images,
                        train_labels,
                        validation_split = 0.2,
                        epochs = 25,
                        batch_size= BATCH_SIZE)
            '''
    st.code(code, language='python')
   
   

