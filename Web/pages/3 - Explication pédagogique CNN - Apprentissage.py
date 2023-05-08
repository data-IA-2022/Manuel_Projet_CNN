import streamlit as st

st.title("Comprendre le mode d'apprentissage des réseaux de neurones convolutifs (CNN).")
st.caption("")
st.caption("")

col11, col12, col13 = st.columns([1,6,1])
   
with col12:        
    st.markdown("![Alt Text](https://i.makeagif.com/media/7-23-2019/q3ItDm.gif)")
    
st.caption("")
st.caption("")
st.caption("Les réseaux de neurones convolutifs (CNN) sont des systèmes d'intelligence artificielle qui pprennent à reconnaître des motifs et des formes dans des images.")
st.caption("Le processus d'apprentissage des CNN se fait en deux étapes. Tout d'abord, le réseau est entraîné sur un ensemble de données d'images pour apprendre à reconnaître les motifs et les caractéristiques des différentes classes d'images (par exemple, chats, chiens, oiseaux).")
st.caption("Pendant cette première étape, le réseau va ajuster les poids de chaque neurone de chaque couche pour minimiser l'erreur de classification entre la sortie prédite par le réseau et la classe réelle de l'image. Cette étape peut prendre beaucoup de temps et nécessite souvent de grandes quantités de données.")
st.caption("")

col11, col12, col13 = st.columns([0.1,6,0.1])
   
with col12:        
    st.markdown("![Alt Text](https://miro.medium.com/v2/resize:fit:640/1*mTTmfdMcFlPtyu8__vRHOQ.gif)")
    
st.caption("Une fois que le réseau a été entraîné, il peut être utilisé pour classifier de nouvelles images. Lorsqu'une nouvelle image est présentée au réseau, celui-ci utilise les caractéristiques qu'il a apprises lors de l'étape d'entraînement pour prédire la classe de l'image.")
st.caption("")
st.caption("")

col11, col12, col13 = st.columns([1,6,1])
   
with col12:        
    st.markdown("![Alt Text](https://miro.medium.com/v2/resize:fit:500/0*61ZaNNpbpMtZLLpZ.)")

st.caption("")
st.caption("")
st.caption("En résumé, le processus d'apprentissage des CNN se fait en deux étapes : l'entraînement sur un ensemble de données d'images pour apprendre à reconnaître les motifs et les caractéristiques des différentes classes d'images, et l'utilisation du réseau entraîné pour classifier de nouvelles images en utilisant les caractéristiques qu'il a apprises. Ce processus permet aux CNN de reconnaître des objets dans des images avec une grande précision.")
