# import libraries
import os, sys
import sys
import random
import math
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image, ImageDraw
from tensorflow.python.framework.versions import VERSION as __version__
import tensorflow as tf
import imgaug

#Cambiamos el Directorio al propio de MASK_RCNN
ROOT_DIR = '/home/saflex/projecto_cleansea/Mask_RCNN-tensorflow2.0'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

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

# define random colors
def random_colors(N):
  np.random.seed(1)
  colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
  return colors

#apply mask to image
def apply_mask(image, mask, color, alpha=0.5):
  for n, c in enumerate(color):
    image[:, :, n] = np.where(mask == 1, image[:, :, n] * (1-alpha) + alpha * c, image[:, :, n])
  return image

#take the image and apply the mask, box, and Label
def display_instances(image, boxes, masks, ids, names, scores):
  n_instances = boxes.shape[0]
  colors = random_colors(n_instances)

  if not n_instances:
    print("NO INSTANCES TO DISPLAY")
  else:
    assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
  for i, color in enumerate(colors):
    if not np.any(boxes[i]):
      continue

    y1, x1, y2, x2 = boxes[i]
    label = names[ids[i]]
    score = scores[i] if scores is not None else None
    caption = "{} {:.2f}".format(label, score) if score else label
    mask = masks[:, :, i]
    
    image = apply_mask(image, mask, color)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
  return image

# Cargamos el archivo json
dataset_dir = "/home/saflex/projecto_cleansea/coco/train_coco_ok"
annotation_json = os.path.join(dataset_dir,"annotations.json")
print(annotation_json)
json_file = open(annotation_json)
coco_json = json.load(json_file)
json_file.close()
print("\nAnotaciones Cargadas\n")
class_names = []
# Añadimos los nombres de las clases usando el metodo de utils.Dataset
source_name = "coco_like"
for category in coco_json['categories']:
    class_id = category['id']
    class_name = category['name']
    if class_id < 1:
        print('Error: Class id for "{}" reserved for the background'.format(class_name))
    else:
        class_names.append(class_name)
print("Nombres Añadidos \n")
print(class_names)

#Mask R-CNN
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

VIDEO_FILE = "/home/saflex/projecto_cleansea/debrisVideo.mp4"
VIDEO_SAVE_DIR = os.path.join('/media/saflex/TOSHIBA EXT/TFG/video_detection', 'savedimgs')
COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_debris_weights1000DA+.h5')
if not os.path.exists(COCO_MODEL_PATH):
  utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(CleanSeaConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 3

config = InferenceConfig()
print(config)
# Create model object in inference mode.
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

video = cv2.VideoCapture(VIDEO_FILE)
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver) < 3 :
  fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
  print('Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}'.format(fps))
else :
  fps = video.get(cv2.CAP_PROP_FPS)
  print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

try:
  if not os.path.exists(VIDEO_SAVE_DIR):
    os.makedirs(VIDEO_SAVE_DIR)
except OSError:
  print ('Error: Creating directory of data')

frames = []
frame_count = 0

while True:
  ret, frame = video.read() 
  if not ret:
    break

# Save each frame of the video to a list
  frame_count += 1
  frames.append(frame)
  print('frame_count :{0}'.format(frame_count))
  
  if len(frames) == 3:
    results = model.detect(frames, verbose=0)
    print('Predicted')
    for i, item in enumerate(zip(frames, results)):
      frame = item[0]
      r = item[1]
      frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
      name = '{0}.jpg'.format(frame_count + i - 3)
      name = os.path.join(VIDEO_SAVE_DIR, name)
      cv2.imwrite(name, frame)
      print('writing to file:{0}'.format(name))
      # Clear the frames array to start the next batch
      frames = []
video.release()