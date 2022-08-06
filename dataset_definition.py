import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
import glob
import pandas as pd
import tensorflow as tf
import shutil

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from PIL import Image

IMG_PATH = "../synthetic_dataset/images/"
JSON_PATH = "../synthetic_dataset/labels/"

TRAIN_PATH = "../synthetic_dataset/train"
TEST_PATH = "../synthetic_dataset/test" 

def extractLabels(img_path=IMG_PATH,json_path=JSON_PATH):
    #Realizamos una lectura de todos los json y extraemos el toppic 'labels' para almacenarla en una variable con todos los labels de todas las imagenes.
    nlabels=[]
    img_names= []

    #Recorremos el folder donde se almacenan los .json
    for file_name in [file for file in os.listdir(json_path) ]:
        if file_name.endswith(".json"):
            with open(json_path + file_name) as json_file:
                content= json.load(json_file)
                #Almacenamos con que imagen va relacionado
                jpegname= content['imagePath']
                #Almacenamos el numero de poligonos que se encuentran dentro de dicho .json
                nshapes= len(content['shapes'])
                #Recogemos los labels de cada uno de los poligonos anteriores
                for topic in range(nshapes):
                    label=content['shapes'][topic]['label']
                    #Añadimos cada label a la lista de labels (excepto las clases con los labels Metal_Chain y WashingMachine ya que no tienen las muestras minimas para poder separarlas) y el path de todas las imagenes
                    if label != 'Metal_Chain' and label != 'WashingMachine':
                        img_names.append( os.path.join(img_path, content['imagePath']))
                        nlabels.append(label)

    #Mostramos todos los labels e imagenes que hemos analizado
    print('Stored Labels:', nlabels)
    print('Images Names:', img_names)

    labels=np.array(nlabels)
    img_names=np.array(img_names)
    return img_names,labels,nlabels

def encodeLabels(labels, nlabels):
    labels, count = np.unique(nlabels, return_counts=True)
    #Los mostramos
    for idx, l in enumerate(labels):
        print(l, ':', count[idx])
    #Inicializamos el Encoder de labes
    le = preprocessing.LabelEncoder()
    #Introducimos los labels en el encoder
    le.fit(nlabels)
    #Aplicamos la codificacion a los labels introducidos y los almacenamos
    Y = le.transform(nlabels)

    #Mostramos los labels codificados
    print(Y)
    print(nlabels)
    return Y

def train_test_split(img_names,Y):
    #Realizamos una division de los datos en 1/5
    skf = StratifiedKFold(n_splits=5)
    train_index, test_index = next( skf.split(img_names, Y) )

    #Almacenamos las imagenes segun donde van a ser movidas
    X_train, X_test = img_names[train_index], img_names[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    #Mostramos las imagenes y la carpeta de destino
    print("Images to be moved to Train Folder: \n", X_train)
    print("\n Images to be moved to Test Folder: \n", X_test)

    #Creamos las carpetas para train y test
    try:
        os.mkdir(TRAIN_PATH)
    except OSError:
        print ("Creation of the directory %s failed" % TRAIN_PATH)
    else:
        print ("Successfully created the directory %s " % TRAIN_PATH)

    try:
        os.mkdir(TEST_PATH)
    except OSError:
        print ("Creation of the directory %s failed" % TEST_PATH)
    else:
        print ("Successfully created the directory %s " % TEST_PATH)

    #Transformamos en listas ambas arrays para evitar imagenes duplicadas en las carpetas
    X_train= list(dict.fromkeys(X_train))
    X_test= list(dict.fromkeys(X_test))
    print(X_train)
    print(X_test)

    #Copiamos las imagenes especificas a la carpeta train
    for f in X_train:
        if os.path.isfile(f):
            #Recorremos el folder donde se almacenan los .json
            for file_name in [file for file in os.listdir(JSON_PATH)]:
                with open(JSON_PATH + file_name) as json_file:
                    content= json.load(json_file)
                    #Almacenamos con que imagen va relacionado
                    jpegname= content['imagePath']
                    full_jpegname= IMG_PATH + jpegname
                    if full_jpegname == f:
                        full_json_name= JSON_PATH + file_name
                        print("Moving TRAIN JSON FILE...",full_json_name,"\n")
                        shutil.copy(full_json_name,TRAIN_PATH)
            print("Moving TRAIN JPG FILE...",f,"\n")
            shutil.copy(f,TRAIN_PATH)


    #Y a la carpeta test
    for f in X_test:
        if (os.path.isfile(f)):
            #Recorremos el folder donde se almacenan los .json
            for file_name in [file for file in os.listdir(JSON_PATH)]:
                with open(JSON_PATH + file_name) as json_file:
                    content= json.load(json_file)
                    #Almacenamos con que imagen va relacionado
                    jpegname= content['imagePath']
                    full_jpegname= IMG_PATH + jpegname
                    if full_jpegname == f:
                        full_json_name= JSON_PATH + file_name
                        print("Moving TEST JSON FILE...",full_json_name,"\n")
                        shutil.copy(full_json_name,TEST_PATH)
            shutil.copy(f,TEST_PATH)
            print("Moving TEST JPG FILE...",f,"\n")

def main():
    img_names,labels,nlabels = extractLabels()
    encoded_labes = encodeLabels(labels,nlabels)
    train_test_split(img_names,encoded_labes)
if __name__ == "__main__":
    main()