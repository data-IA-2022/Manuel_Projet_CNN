import streamlit as st
import cv2

# Création de 7 onglets
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Sens visuel humain", 
                                                    "Model artificiel", 
                                                    "La convolution",
                                                    "Le pooling",
                                                    "Fct d'activation",
                                                    "Flatten",
                                                    "Fully connected net"])

# Onglet 1 : Sens visuel humain
with tab1:
    st.title("Sens visuel humain.")
    
    col11, col12, col13 = st.columns([1,6,1])
   
    with col12:             
        st.image(cv2.imread('.\Images\Vision_Humain.jpg'), 
              caption='Système visuel ganglionnaires')
    
    st.text('''
            Le système visuel humain est responsable de la façon dont nous voyons le monde 
            autour de nous. Quand nous regardons quelque chose, la lumière qui vient de 
            cet objet entre dans nos yeux et passe à travers des parties spéciales appelées 
            cornées, pupilles et lentilles. Cela aide à focaliser la lumière sur la partie 
            arrière de l'œil, qui est appelée la rétine.
            
            La rétine contient des cellules spéciales appelées photorécepteurs, qui sont 
            sensibles à la lumière et qui envoient des signaux électriques au cerveau. 
            Ces signaux électriques voyagent à travers des nerfs qui se rejoignent en une 
            grosse corde appelée nerf optique. Le nerf optique envoie ensuite les signaux au 
            cerveau pour être traités.
            
            Le cerveau utilise ces signaux pour construire une image complète de ce que nous 
            regardons. Il utilise des parties spéciales du cerveau appelées cortex visuel 
            pour aider à comprendre et à interpréter les images. Le cerveau peut également 
            aider à remplir les parties manquantes de l'image ou les zones floues pour que 
            nous puissions voir une image claire.
            
            Le système visuel humain est très complexe et il est en constante évolution. 
            Nous en apprenons toujours plus sur la façon dont il fonctionne et sur les 
            différentes parties du cerveau qui y sont impliquées. Mais grâce à ce système, 
            nous sommes capables de voir les merveilles du monde autour de nous et de 
            comprendre ce que nous regardons.''')
   
    st.caption("")
    st.caption("")

    col11, col12, col13 = st.columns([1,6,1])
   
    with col12:             
        st.image(cv2.imread('.\Images\Capture_triangle.png'), 
                  caption='Caractéristiques de bord et de contours')
        
# Onglet 2 : Transposition informatique du système visuel humain
with tab2:
    st.title("Transposition informatique du système visuel humain.")
    
    col11, col12, col13 = st.columns([0.5,6,0.5])
   
    with col12:        
        st.image(cv2.cvtColor(cv2.imread('.\Images\conv.png'), cv2.COLOR_BGR2RGB))
        st.image(cv2.cvtColor(cv2.imread('.\Images\Typical_cnn.jpg'), cv2.COLOR_BGR2RGB),
                 caption="L’architecture d'un cnn")
        
    st.text('''
            Un CNN est une machine qui passe une image à travers plusieurs étapes pour 
            la transformer en une prédiction ou une classification.
            
            La première étape est l'étape de convolution, où l'image est 'scannée' par 
            des filtres qui recherchent des formes et des motifs spécifiques. Par exemple, 
            un filtre peut rechercher des bords dans l'image. Cette étape aide le CNN 
            à comprendre les caractéristiques de l'image.
            
            Ensuite, l'étape de pooling réduit la taille de l'image en gardant seulement 
            les caractéristiques les plus importantes. Cela aide le CNN à traiter plus 
            rapidement les images de grande taille.
            
            Après cela, il y a une étape de flattening où l'image est transformée en 
            un vecteur unidimensionnel pour que le CNN puisse la traiter plus facilement.
            
            Ensuite, les vecteurs sont passés dans plusieurs couches denses, qui sont des 
            couches où des calculs mathématiques sont effectués pour transformer les vecteurs 
            et les combiner.
            
            Finalement, le CNN donne une prédiction ou une classification de l'image basée 
            sur les transformations effectuées dans les couches précédentes.
            
            
            En termes de comparaison avec le système visuel animal, on peut dire que les 
            couches de convolution correspondent aux cellules du cortex visuel qui détectent 
            les caractéristiques visuelles de base comme les bords, les coins et les couleurs. 
            Les couches de pooling correspondent à la façon dont le système visuel animal 
            réduit la taille de l'image en se concentrant sur les zones les plus importantes. 
            Les couches entièrement connectées peuvent être comparées aux zones du cerveau qui 
            utilisent ces caractéristiques visuelles pour identifier et classifier les objets 
            dans l'environnement.''')
    
    st.caption("")
    st.caption("")
    
    col11, col12, col13 = st.columns([1,6,1])
   
    with col12:        
        st.image(cv2.cvtColor(cv2.imread('.\Images\cnn_2.jpg'), cv2.COLOR_BGR2RGB), 
                 caption='Extraction de caractéristiques multi-layers')

# Onglet 3 : Convolution         
with tab3:
    st.title("Principe et buts de la convolution. ")

    col11, col12, col13 = st.columns([0.2,6,0.2])

    with col12:        
        st.markdown("![Alt Text](https://miro.medium.com/v2/format:jpg/resize:fit:640/1*h01T_cugn22R2zbKw5a8hA.gif)")
    
    st.text('''
            Les CNN utilisent des filtres pour détecter les caractéristiques visuelles 
            de base dans les images, telles que les bords et les coins. Les filtres sont 
            appliqués à l'image en effectuant une opération de convolution, qui permet 
            de calculer la réponse de chaque filtre à chaque position de l'image.''')
    st.caption("")
    st.caption("")
   
    col11, col12, col13 = st.columns([1,6,1])
    
    with col12:        
        st.image(cv2.cvtColor(cv2.imread('.\Images\Capture_lena.png'), cv2.COLOR_BGR2RGB),
                 caption='Extraction de caractéristiques')
    
    st.text('''
            Un CNN peut être considérée comme une approximation informatique de la façon 
            dont le cerveau humain utilise des cellules ganglionnaires pour détecter les 
            caractéristiques visuelles de base dans les images.''')
  
# Onglet 4 : Pooling  
with tab4:
    st.title("Principe et buts du pooling.")
    st.caption("")
    st.caption("")
    
    col11, col12, col13 = st.columns([0.2,6,0.2])

    with col12:        
        st.markdown("![Alt Text](https://victorzhou.com/ac441205fd06dc037b3db2dbf05660f7/pool.gif)")
        
    st.text('''
            Le pooling est une technique utilisée dans les réseaux de neurones pour réduire
            la taille des images ou des données. Cela permet de simplifier l'information et 
            de rendre le traitement plus rapide et plus efficace.
            
            Dans le contexte des réseaux de neurones convolutionnels (CNN), le pooling est 
            souvent utilisé après une couche de convolution. La couche de convolution aide 
            à extraire des caractéristiques d'une image en la filtrant à travers des matrices 
            de convolutions. Ensuite, le pooling peut être appliqué pour réduire la taille 
            de l'image résultante.''')
    st.caption("")
    st.caption("")
    
    col11, col12, col13 = st.columns([1,6,1])

    with col12:   
        st.image(cv2.cvtColor(cv2.imread('.\Images\Capture_pool.png'), cv2.COLOR_BGR2RGB),
                 caption="Réduction de dimentions")
    st.text('''
            Le pooling est souvent réalisé en utilisant une matrice de petite taille, 
            comme un carré de 2x2 ou 3x3. Cette matrice est appliquée à l'image en la faisant 
            glisser dessus et en sélectionnant une seule valeur pour chaque région recouverte 
            par la matrice. Par exemple, si nous utilisons une matrice de 2x2, nous pourrions 
            prendre la valeur la plus élevée de chaque groupe de quatre pixels. Ce processus 
            est appelé max pooling.
            
            Le pooling peut également être utilisé pour faire de la moyenne des valeurs 
            de chaque groupe de pixels, appelé average pooling. Cette technique est utile pour
            réduire la taille des images tout en conservant une certaine information sur 
            l'image d'origine.''')
    
    col11, col12, col13 = st.columns([1,6,1])

    with col12:   
        st.image(cv2.cvtColor(cv2.imread('.\Images\Pooling-process-of-input-feature-that-illustrates-the-drawbacks-of-max-pooling-and.png'), cv2.COLOR_BGR2RGB),
                 caption="Processus de pooling des caractéristiques d'entrée")
        
    st.text('''
            En résumé, le pooling est une technique de réduction de la taille de l'image
            ou des données qui permet de simplifier l'information et de rendre le traitement 
            plus rapide et efficace. C'est une étape importante dans les réseaux de neurones 
            convolutionnels pour aider à extraire des caractéristiques d'une image et 
            à les utiliser pour des tâches telles que la reconnaissance d'objets ou la 
            classification d'images.''')

# Onglet 5 : Fonction d'activation       
with tab5:
    st.title("Principe et buts d'une fonction d'activation.")
    st.caption("")
    st.caption("")
    
    col11, col12, col13 = st.columns([0.8,6,0.1])

    with col12:        
        st.markdown("![Alt Text](https://storage.googleapis.com/kaggle-media/learn/images/a86utxY.gif)")
 
    st.text('''
            La fonction d'activation est une méthode utilisée dans les réseaux de 
            neurones artificiels pour aider à décider si une information doit être 
            transmise au neurone suivant ou non.
            
            C'est un peu comme le filtre que l'on utilise pour séparer le café des 
            grains de café. Si les grains sont trop grands, le filtre ne les laisse 
            pas passer et le café est plus clair et plus pur. De même, la fonction 
            d'activation aide à filtrer les informations en ne laissant passer que 
            les informations importantes pour la tâche que l'on veut accomplir.
            
            Il y a différentes fonctions d'activation, mais l'une des plus couramment 
            utilisées est la fonction ReLU, qui permet de transformer les valeurs d'entrée 
            en valeurs de sortie selon une certaine règle.''')
   
    st.caption("")
    st.caption("")
    
    col11, col12, col13 = st.columns([0.1,6,0.1])
    
    with col12:   
        st.image(cv2.cvtColor(cv2.imread('.\Images\slide_10.jpg'), cv2.COLOR_BGR2RGB))
    
    st.text('''
            En somme, la fonction d'activation est un outil important qui permet de mieux 
            comprendre et de mieux utiliser les réseaux de neurones pour accomplir des tâches 
            utiles comme la reconnaissance d'image ou la prédiction de résultats.''')
  
# Onglet 6 : Couche Flatten    
with tab6:
    st.title("Principe et buts de la couche Flatten.")
    st.caption("")
    st.caption("")
    
    col11, col12, col13 = st.columns([0.1,6,0.1])
    
    with col12:   
        st.image(cv2.cvtColor(cv2.imread('.\Images\Capture_Flatten.png'), cv2.COLOR_BGR2RGB))
    
    st.text('''
            La couche Flatten est une couche utilisée dans les réseaux de neurones pour 
            transformer les données en une forme plus simple et plus facile à manipuler.
            
            C'est un peu comme lorsqu'on plie une feuille de papier pour la ranger dans 
            une enveloppe. Si la feuille est grande et compliquée, elle ne rentre pas 
            dans l'enveloppe. Mais si on la plie bien, elle rentre facilement. La couche 
            Flatten fait quelque chose de similaire : elle prend les données de la couche 
            précédente, qui peuvent être complexes et multidimensionnelles, et les 
            transforme en une simple liste de données à une dimension.
            
            Par exemple, si on veut utiliser un réseau de neurones pour classer des images 
            de chats et de chiens, la couche Flatten peut prendre les pixels de l'image 
            (qui sont organisés en une grille de deux dimensions) et les transformer en une 
            seule liste de pixels, ce qui facilite le travail de classification.
            
            En résumé, la couche Flatten est une étape importante dans la construction 
            de réseaux de neurones, car elle permet de transformer les données complexes en 
            une forme plus simple et plus facile à manipuler pour accomplir des tâches comme 
            la reconnaissance d'image.''')

# Onglet 7 : Couche Fully Connected
with tab7:
    st.title("Principe et buts du Fully connected net.")
    st.caption("")
    st.caption("")
    
    col11, col12, col13 = st.columns([0.1,6,0.1])

    with col12:        
        st.markdown("![Alt Text](https://thumbs.gfycat.com/ContentDarlingCub-size_restricted.gif)")
        
    st.text('''
            Un Fully Connected Network est un type de réseau de neurones artificiels où chaque 
            neurone de chaque couche est connecté à tous les neurones de la couche précédente
            et de la couche suivante.
            
            Dans le cas des réseaux de neurones convolutifs (CNN), les couches de convolution 
            et de pooling sont utilisées pour extraire les caractéristiques principales des 
            images, mais il reste encore une étape importante à accomplir : la classification.
            
            C'est là qu'intervient le Fully Connected Network. Il s'agit d'une couche qui prend 
            les caractéristiques extraites par les couches de convolution et de pooling 
            et les utilise pour classifier l'image. En d'autres termes, cette couche va 
            déterminer si l'image représente un chat, un chien ou autre chose.
            
            Imaginez que vous regardiez une photo d'un animal de compagnie. Les couches de 
            convolution et de pooling ont déjà extrait les principales caractéristiques de 
            l'image, telles que la forme et la couleur de l'animal. La couche Fully Connected
            Network va maintenant utiliser ces caractéristiques pour déterminer si l'image 
            représente un chat ou un chien.
            
            En résumé, le Fully Connected Network est une couche importante des réseaux de
            neurones convolutifs, qui permet de classifier les images après que les 
            caractéristiques principales ont été extraites. C'est grâce à cette couche que les 
            CNN sont capables de reconnaître des objets dans des images avec une grande
            précision.''')
            