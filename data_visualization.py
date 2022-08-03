import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
ia.seed(1)
import os
import json
from tqdm import tqdm
import numpy as np
import imageio
import time
import cv2

AUG_PATH = "../data_augmented/"
N_AUG = 1

IMG_DATASET = "D:/Cleansea/cleansea_dataset/Dataset/DebrisImages"
LABELS_DATASET = "D:/Cleansea/cleansea_dataset/Dataset/Annotations"

LABELS = ["background","Can","Squared_Can","Wood","Bottle","Plastic_Bag","Glove","Fishing_Net","Tire","Packaging_Bag","WashingMachine","Metal_Chain","Rope","Towel","Plastic_Debris","Metal_Debris","Pipe","Shoe","Car_Bumper","Basket"]
COLOR_CODE = [(255,0,0),(0,255,0),(0,0,255),(255,155,0),(255,155,155),(155,255,0),(155,255,155),(155,0,255),(155,155,255),(255,255,0),(0,255,255),(255,0,255)]
# COLOR_CODE = [(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0)]

import random

def main():
    data_augment()

def data_augment():
    if not os.path.exists(AUG_PATH):
        os.mkdir(AUG_PATH)
    for img_p in tqdm(os.listdir(IMG_DATASET)):
        image, mask = load_image_label(img_p)
        if not image is None and not mask is None:
                times = 0
                while(times<N_AUG):
                    s_time = time.time()
                    image = cv2.resize(image, dsize=(512, 512))
                    #print(mask.shape)
                    mask = cv2.resize(mask, dsize=(512, 512)).astype(np.uint8)
                    augmenter = obtain_augmenter(image.shape,mask)
                    #print(image.shape)
                    #print(mask.shape)
                    images_aug, segmaps_aug = augmenter(image=image, segmentation_maps=SegmentationMapsOnImage(mask, shape=image.shape))
                    e_time = time.time()
                    #print("generate")
                    #print(e_time-s_time)
                    s_time = time.time()
                    grid_image = draw_result(image, SegmentationMapsOnImage(mask, shape=image.shape), images_aug,segmaps_aug)
                    if not os.path.exists(AUG_PATH):
                        os.mkdir(AUG_PATH)
                    imageio.imwrite(os.path.join(AUG_PATH,img_p.replace(".jpg",f"_{times}.jpg")), grid_image)
                    e_time = time.time()
                    #print("save")
                    #print(e_time-s_time)
                    times+=1

def load_image_label(img_p,labels_path= LABELS_DATASET):
    ann_path = img_p.replace(".jpg",".json")
    annotation = os.path.join(labels_path, ann_path)
    data = load_json(annotation)
    #print(data['shapes'][0])
    labels = (data['shapes'])

    if len(labels) > 0:
        img_orig = cv2.cvtColor(cv2.imread(os.path.join(labels_path,img_p)), cv2.COLOR_BGR2RGB)

        shape_mask = np.zeros((img_orig.shape[0], img_orig.shape[1]) + (max(1, 1),)).astype(np.uint8)
        shape_mask_aux = np.zeros((img_orig.shape[0], img_orig.shape[1]))
        for label in labels:
            shape_mask_aux = load_mask(annotation, shape_mask_aux)
            shape_mask[:, :, 0] = np.logical_or(shape_mask[:, :, 0], shape_mask_aux)
        
        return img_orig, shape_mask
    else:
        return None, None

def load_mask(label, shape_mask):

    data = load_json(label)
    shapes= data['shapes']
    for shape in shapes:
        shape_mask = cv2.fillPoly(shape_mask, [np.array(shape['points'], dtype=np.int32)], color=1)
    return shape_mask

def load_json(json_path):
    # load label file
    with open(json_path) as f:
        label = f.read()
    # convert label to json
    return json.loads(label)

def obtain_augmenter(shape, mask):
    
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order
    
    ###################################################
    # Modification for visualization
    ###################################################
    seq= iaa.Sequential()
    
    return seq
    
def draw_result(image, segmap, image_aug,segmap_aug):
    cells = []
    cells.append(image)                # column 1
    #cells.append(image_aug)                                     # column 2
    cells.append(segmap_aug.draw_on_image(image_aug)[0])        # column 3

    # Convert cells to a grid image and save.
    grid_image = ia.draw_grid(cells, cols=2)

    return grid_image

if __name__ == "__main__":
    main()