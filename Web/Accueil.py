import streamlit as st
# import pandas as pd
# from joblib import load
import index
# from tensorflow.keras.models import load_model

lst = ["Image 1", "Image 2", "Images de test mal classées", "Toutes les images de test"]


img1 = index.load_image()
img2 =index.load_image_2()
images, labels, predict_list, predict_reject_list, lst_classes, history =index.load_test_images('\CNN V2.4 0.1 91%')
img=None

if 'texte_bouton_1' not in st.session_state:
    st.session_state['texte_bouton_1'] = "Suivant"

if 'texte_bouton_2' not in st.session_state:
    st.session_state['texte_bouton_2'] = "Précédent"

if 'expliaction_buton' not in st.session_state:
    st.session_state['expliaction_buton'] = 0
    
if 'model' not in st.session_state:
    st.session_state['model'] = index.load_my_model()
    
if 'model_graf' not in st.session_state:
    st.session_state['model_graf'] = index.load_model_graf(st.session_state['model'])
    
if 'index_image' not in st.session_state:
    st.session_state['index_image'] = 10
    
if 'index_bad_image' not in st.session_state:
    st.session_state['index_bad_image'] = 10

if 'selector_type_image' not in st.session_state:
    st.session_state['selector_type_image'] = 2

if 'predict_image' not in st.session_state:    
    st.session_state['img']=img
     
def prediction(img, images, model):
    return index.prediction(img, images, model)
    
def prediction_2(img, model):
    return index.prediction_2(img, model)
    
# col11, col12, col13 = st.columns([1,6,1])
   
# with col12:        
#     st.title("Classification d'images - Deep Learning & CNN")
    
st.markdown("<h1 style='text-align: center; color: grey;'>Classification d'images - Deep Learning & CNN</h1>", unsafe_allow_html=True)

# st.caption("")
# st.caption("")

col11, col12, col13 = st.columns([1,6,1])
   
with col12:        
    st.markdown("![Alt Text](https://res.cloudinary.com/nuxeo/image/upload/f_auto,w_600,q_auto,c_scale/v1//blog/cat-or-dog-ai.gif)")
    
st.caption("Développer une application qui permet de détecter automatiquement des images d'animaux Chiens et Chats.")
st.caption("L'utilisateur doit pouvoir uploader une photo et l'application doit préciser de quel animal il s'agit ainsi que la probabilité de la classification.")
st.caption("Le classifieur sera développé avec Keras.")
    
    
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Critères de performance", 
                                                "Contexte du projet",
                                                "Modalités pédagogiques",
                                                "Modalités d'évaluation",
                                                "Livrables",
                                                "Ressource(s)"])

with tab6:
    st.markdown("https://keras.io/examples/vision/image_classification_from_scratch/")
    st.markdown("https://poloclub.github.io/cnn-explainer/")
    st.markdown("https://keras.io/about/")
    st.markdown("https://stanford.edu/~shervine/l/fr/teaching/cs-230/pense-bete-reseaux-neurones-convolutionnels")
    
with tab2:
    st.caption("En tant que développeur en IA,")
    st.caption("    - Analyser le besoin")
    st.caption("    - Lister les tâches à réaliser et estimer le temps de réalisation")
    st.caption("    - Réaliser un planning prévisionnel du projet")
    st.caption("    - Développer l'IA")
    st.caption("    - Mettre en place l'application (web, IA)")
    st.caption("    - Livrer l'application au commanditaire")
    
with tab3:    
    st.caption("    - Consulter en priorité les ressources données dans le projet. Il existe bien sûr d'autres ressources utiles")
    st.caption("    - Projet en individuel")
    st.caption("    - Collaboration et échanges possibles (voir recommandés) entre les étudiants")
    st.caption("    - Projet en complète autonomie, pas ou peu de support des formateurs (car réalisé pendant la période de stage)")
    st.caption("Bonus (à faire seulement si toutes les étapes ont déjà été réalisées) :")
    st.caption("    - Utiliser la data augmentation pour améliorer les performances de l'algorithme")
    st.caption("    - Améliorer les performances de l'algorithme en utilisant un modèle pré-entrainé, grâce au Transfer Learning")
    st.caption("    - Augmenter les usages de l'application par la classification d'autres animaux (cheval, oiseau, etc) et/ou d'objets en utilisant un modèle pré-entrainé, grâce au Transfer Learning")
    st.caption("    - Augmenter les fonctionnalités de l'application par ajout de composants supplémentaires que vous jugerez utiles")
    st.caption("    - Tout autre point que vous jugerez utile")

with tab4:  
    st.caption("    - Démo de l'application")
    st.caption("    - Présentation synthétique des CNN : objectif, fonctionnement, forces, faiblesses, cas d'usage, etc.")
    
with tab5:
    st.caption("    - Des scripts Python fonctionnels et commentés, déposés sur Github")
    st.caption("    - Une application Streamlit qui comporte au moins 3 onglets :")
    st.caption(" 1. Page d'accueil")
    st.caption(" 2. Page de chargement de l'image et de restitution des prédictions")
    st.caption(" 3. Page d'explication pédagogique des réseaux de neurones CNN")
    
with tab1:
    st.caption(" - Développement du CNN avec Keras")
    st.caption(" - Obtenir une accuracy >90% lors de la validation après apprentissage sur 25 epochs.")
    st.caption(" - Présenter une application fonctionnelle et qui répond aux exigences.")
    st.caption(" - Etre en mesure d'expliquer le CNN à ma grand-mère.")
    