# Manuel_Projet_CNN

images sur https://www.microsoft.com/en-us/download/details.aspx?id=54765


Les modeles sont générés dans les répertoires suivant.
	-	CNN_V2_4_0_1_91 (CNN Custom plojet pincipal)
	-	CNN_V2_4_0_1_91_2 class_aug (CNN Custom Bonus data augmentaion)
	-	CNN_V2_4_0_1_91_3_class (CNN Custom Bonus 3 classes avec images Chien et chats 2*12500 images et 2376 images cheveaux)
	-	CNN_V2_4_0_1_91_3_class_aug_horse (CNN Custom Bonus 3 classes avec images Chien et chats 2*12500 images et augmentation à 11880 images cheveaux)
	-	MobileNetV2 --> Fine tunning
								--> MobileNetV2 Imagenet (CNN MobileNetV2 Bonus alpha=1)
								--> MobileNetV2 Imagenet 0.35 (CNN MobileNetV2 Bonus alpha=0.35)
								--> MobileNetV2 Imagenet 3 class (CNN MobileNetV2 Bonus alpha=1 avec 3éme classe avec 11880 images)
								--> MobileNetV2 Imagenet class aug (CNN MobileNetV2 Bonus alpha=1 avec data augmentation)
								--> MobileNetV2Imagenet 0.35 3 class (CNN MobileNetV2 Bonus alpha=0.35 avec 3éme classe avec 11880 images)
								--> MobileNetV2Imagenet 0.35 aug (CNN MobileNetV2 Bonus alpha=0.35 avec data augmentation)
					--> Training
								--> MobileNetV2 (CNN MobileNetV2 Bonus alpha=1)
								--> MobileNetV2 0.35 (CNN MobileNetV2 Bonus alpha=0.35)
								--> MobileNetV2 64x64 - 5 (CNN MobileNetV2 Bonus alpha= 5 est scaling des images à 64x64)
					--> Transfere learning
								--> MobileNetV2 160x160 (CNN MobileNetV2 Bonus alpha=1)
								--> MobileNetV2 160x160 - 0.35 (CNN MobileNetV2 Bonus alpha=0.35)
								--> MobileNetV2 224x224 (CNN MobileNetV2 Bonus alpha=1 et scaling des images à 224x224)
								-->	MobileNetV2 224x224 - 0.35 (CNN MobileNetV2 Bonus alpha=0.35 est scaling des images à 224x224)
	-	Resnet50	--> CNN Resnet50 fine (CNN Resnet50 Bonus)
					--> CNN Resnet50 train (CNN Resnet50 Bonus)
					--> CNN Trans Learn 244x244 batch_5 (CNN Resnet50 Bonus, scaling des images à 224x224)
					--> CNN Trans Learn 160x160 aug (CNN Resnet50 Bonus avec data augmentation)

Les fichiers CNN_net.py sont les fichier d'entrainement des CNN. --> génère les fichiers history.pickle et *.H5
Les fichiers evaluate_validation_dataset.py généres les scores, matrices de confusion, et ROC --> fichiers score.pickle, confusion_matrix.pickle, roc_fpr.pickle et roc_tpr.pickle

Ficher CNN_load_dataset --> génère des fichier intermédière dans Web dataset_*.pickle
Ficher "CNN_load_dataset 3 class" --> génère des fichier intermédière dans Web dataset_3_*.pickle
Ficher "CNN_load_dataset 3 class aug" --> génère des fichier intermédière dans Web dataset_3_aug_horse*.pickle
Ficher "CNN augmentation du nombre image Horses" --> génère des images data augmentées dans le répertoire "PetImages 3 class augmente"

L'application Streamlit se trouve dans le répertoire Web (plojet pincipal)
Elle se compose des fichier de démarage "Acceil.py" et index.py
Les formulaires de l'application se trouvent dans le répertoire "pages"
Les images utiles à l'application soit elle sont dans le répertoire "images" ou directement accéssible par intenet.
Une connection à Internet est indispensable
