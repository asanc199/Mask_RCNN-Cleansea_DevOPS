from audioop import add
import os
import sys
from numpy import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
from PIL import Image, ImageDraw
from tensorflow.python.framework.versions import VERSION as __version__
import tensorflow as tf
from imgaug import augmenters as iaa
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm, datasets
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

#Cambiamos el Directorio al propio de MASK_RCNN
ROOT_DIR = './'
#ROOT_DIR = '/home/saflex/Projecto_CleanSea/Mask_RCNN/Mask_RCNN-master'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
############################################################
#  Configuracion
############################################################

class CleanSeaConfig(Config):
    """
    Configuracion para el entrenamiento con CleanSea Dataset.
    """

    # Nombre de la configuracion
    NAME = "debris"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Numero de clases + el background
    NUM_CLASSES = 1 + 19  # Cleansea tiene 19 clases

    # Salta las detecciones con <50% de seguridad
    DETECTION_MIN_CONFIDENCE = 0.5
    

config= CleanSeaConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Directorio perteneciente a MASK-RCNN
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Dataset
############################################################
class CleanSeaDataset(utils.Dataset):
    def load_data(self, dataset_dir, subset):
        # Train or validation dataset?
        assert subset in ["train_coco", "test_coco"]
        dataset_dir = os.path.join(dataset_dir, subset)
        print(dataset_dir)

        # Cargamos el archivo json
        annotation_json = os.path.join(dataset_dir,"annotations.json")
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        print("\nAnotaciones Cargadas\n")

        # Añadimos los nombres de las clases usando el metodo de utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" reserved for the background'.format(class_name))
            else:
                self.add_class(source_name, class_id, class_name)
        print("Nombres Añadidos \n")

        # Almacenamos las anotaciones
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        print("Anotaciones Almacenadas\n")

        # Almacenamos las imagenes y las añadimos al dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.join(dataset_dir, image_file_name)
                image_annotations = annotations[image_id]
                
                # Añadimos la imagen usando el metodo de utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
        print("Imagenes añadidas al Dataset\n")

    def load_mask(self, image_id):
        """ Carga la mascara de instancia para la imagen dada
        MaskRCNN espera mascaras en forma de mapa de bits (altura, anchura e instancias)
        Argumentos:
            image_id: El ID de la imagen a la que vamos a cargar la mascara
        Salida:
            masks: Una cadena booleana con estructura (altura, anchya y la cuenta de instancias) con una mascara por instancia
            class_ids: Una cadena de 1 dimension de clase ID de la instancia de la mascara """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Matriz de Confusión
############################################################
def confusion_matrix(y_test,y_pred):
    # import some data to play with
    class_names = ['background', 'Can', 'Squared_Can', 'Wood', 'Bottle', 'Plastic_Bag', 'Glove', 'Fishing_Net', 'Tire', 'Packaging_Bag', 'WashingMachine', 'Metal_Chain', 'Rope', 'Towel', 'Plastic_Debris', 'Metal_Debris', 'Pipe', 'Shoe', 'Car_Bumper', 'Basket']
    # Plot non-normalized confusion matrix
    title = "Normalized confusion matrix"

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true= y_test,
        y_pred= y_pred,
        normalize='true',
        include_values=True,
        cmap=plt.cm.Blues,
        xticks_rotation='vertical',
        values_format='.2f'
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.show()

############################################################
#  Evaluacion
############################################################
class InferenceConfig(CleanSeaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_debris_weights1000DA5Heads.h5")
model_path = os.path.join(MODEL_DIR, "debris20220706T2113/mask_rcnn_debris_0104.h5")
#model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Training dataset.
dataset_train = CleanSeaDataset()
print("Configuracion para dataset_train cargada\n")
dataset_train.load_data("./CocoFormatDataset","train_coco")
print("Dataset Inicializado Correctamente\n")
dataset_train.prepare()
print("Preparacion del Dataset Completada\n")

# Validation dataset
dataset_test = CleanSeaDataset()
print("Configuracion para dataset_test cargada\n")
dataset_test.load_data("./CocoFormatDataset", "test_coco")
print("Dataset Inicializado Correctamente\n")
dataset_test.prepare()
print("Preparacion del Dataset Completada\n")

#Configuramos el path para los archivos .json
JSON_PATH = "./Dataset/test/json"


#Realizamos una lectura de todos los json y extraemos el toppic 'labels' para almacenarla en una variable con todos los labels de todas las imagenes.
nlabels=[]
img_names= []

#Recorremos el folder donde se almacenan los .json
for file_name in [file for file in os.listdir(JSON_PATH)]:
  with open(JSON_PATH + "/" + file_name) as json_file:
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
        nlabels.append(label)

#Mostramos todos los labels e imagenes que hemos analizado
#print('Stored Labels:', nlabels)

class_names=np.array(nlabels)
img_names=np.array(img_names)

############################################################
#  Deteccion Deseada vs Obtenida
############################################################
# Test on a random training image
image_id = 138
print(f"Image {image_id} to process...")
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inference_config, 
                           image_id)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))
plt.show()
# Resultados de la deteccion procesada por el modelo
print("Detection done by trained model...")
results = model.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_test.class_names, r['scores'], ax=get_ax(),figsize=(8,8))
plt.show()
############################################################
#  Curva de Precision-Recall
############################################################
# Draw precision-recall curve
print("Precision-recall curve")
AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
visualize.plot_precision_recall(AP, precisions, recalls)
plt.show()
############################################################
#  Precision del Modelo
############################################################
print("Calculating mAP...")

#ground-truth and predictions lists
gt_tot = np.array([])
pred_tot = np.array([])
#mAP list
mAP_ = []
compare_images = []

"""
# Comparacion con estudios anteriores (Se escogen determinadas clases)
for image_id in dataset_test.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config, image_id)
    classes = gt_class_id
    added = False
    for object in classes:
        if object == 1 and added==False:
            compare_images.append(image_id)
            added = True
        #elif object == 3 and added==False:
        #    compare_images.append(image_id)
        #    added = True
        elif object == 4 and added==False:
            compare_images.append(image_id)
            added = True
        elif object == 5 and added==False:
            compare_images.append(image_id)
            added = True
        elif object == 7 and added==False:
            compare_images.append(image_id)
            added = True
        elif object == 9 and added==False:
            compare_images.append(image_id)
            added = True
        #elif object == 11 and added==False:
        #    compare_images.append(image_id)
        #    added = True
        elif object == 12 and added==False:
            compare_images.append(image_id)
            added = True
        elif object == 14 and added==False:
            compare_images.append(image_id)
            added = True
        #elif object == 15 and added==False:
        #    compare_images.append(image_id)
        #    added = True
        elif object == 16 and added==False:
            compare_images.append(image_id)
            added = True
        elif object == 17 and added==False:
            compare_images.append(image_id)
            added = True
print(compare_images)
"""

#compute gt_tot, pred_tot and mAP for each image in the test dataset
for image_id in dataset_test.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config, image_id)
    info = dataset_test.image_info[image_id]

    # Run the model
    results = model.detect([image], verbose=1)
    r = results[0]
    
    #compute gt_tot and pred_tot
    gt, pred = utils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
    gt_tot = np.append(gt_tot, gt)
    pred_tot = np.append(pred_tot, pred)
    
    #precision_, recall_, AP_ 
    AP_, precision_, recall_, overlap_ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
    
    mAP_.append(AP_)

gt_tot=gt_tot.astype(int)
pred_tot=pred_tot.astype(int)
print(f"Test Dataset: {dataset_test.class_names}")
#print("ground truth list: ",gt_tot)
#print("predicted list: ",pred_tot)

norm_detections = []
norm_gt = []
for i in gt_tot:
    norm_gt.append(dataset_test.class_names[i])
for i in pred_tot:
    norm_detections.append(dataset_test.class_names[i])

#print(f"Filtered GT: {norm_gt}")
#print(f"Filtered Detections: {norm_detections}")
#print(f"Accuracy list {mAP_}")

print("mAP: ", np.mean(mAP_))

#save the vectors of gt and pred
save_dir = "output"
gt_pred_tot_json = {"gt_tot" : gt_tot, "pred_tot" : pred_tot}
df = pd.DataFrame(gt_pred_tot_json)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
df.to_json(os.path.join(save_dir,"gt_pred_test.json"))

############################################################
#  Matriz de Confusion
############################################################
# Grid of ground truth objects and their predictions
print("Confusion Matrix")
confusion_matrix(norm_gt,norm_detections,)