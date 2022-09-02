import os
from pickletools import uint8
from sklearn import datasets
from tqdm import tqdm
import numpy as np
import cv2
import random
from random import randrange, sample
from scipy import ndimage,misc
import json
import math
import data_visualization
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imageio

import base64

import matplotlib.pyplot as plt

random.seed(1)

ROT = True
SCALE = True
TRANS = True
TRY = 3
OVERLAP = 40

AUG_PATH = "synthetic_data"

#------------------------------------------------------------------------------
def rotate(xy, angle, origin=(0,0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(math.radians(angle))
    sin_rad = math.sin(math.radians(angle))
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return [qx, qy]

#------------------------------------------------------------------------------
def patch_json(name,json_path,angle,x,y,rs,patch,new_img,path_to_new_img):
    """
    Creates a JSON Annotation file for the synthetic image
    """
    data = json.load( open(json_path) )
    assert len(data["shapes"]) != 0, f"Error: no shape in {json_path}"

    p_h = float(patch.shape[0])
    p_w = float(patch.shape[1])
    assert p_h == data["imageHeight"]
    assert p_w == data["imageWidth"]

    # Adding these fields to the object images that do not have it:
    if 'line_color' not in data['shapes'][0].keys(): data['shapes'][0]['line_color'] = None
    if 'fill_color' not in data['shapes'][0].keys(): data['shapes'][0]['fill_color'] = None


    dsp_pts = []
    i=0
    for shape in range(len(data["shapes"])):
        for pt in data["shapes"][shape]["points"]:
            #print(f"Original Point: {pt}")
            new_pt = [pt[0],pt[1]]

            if ROT:                            #Apply rotation
                rot_og = (p_w/2, p_h/2)
                new_pt = rotate(new_pt,angle,rot_og)

                M = cv2.getRotationMatrix2D(rot_og, angle, 1.0)     # AJUSTE...
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((p_h * sin) + (p_w * cos))
                nH = int((p_h * cos) + (p_w * sin))
                new_pt[0] += (nW / 2) - rot_og[0]
                new_pt[1] += (nH / 2) - rot_og[1]

                #print(f"Rotated Point: {new_pt}")

            if SCALE:                       #Apply resize
                #print(f"Resize: {rs}")
                rs_x = ((p_w*rs)*new_pt[0])/p_w
                rs_y = ((p_h*rs)*new_pt[1])/p_h
                new_pt = [rs_x,rs_y]
                #print(f"Resized Point: {new_pt}")

            if TRANS:                       #Apply Displacement
                new_pt = [float(new_pt[0]+x),float(new_pt[1]+y)]
                #print(f"Displaced Point: {new_pt}")

            dsp_pts.append(new_pt)
            i+=1

        new_data,shp = {},{}
        #print(data)
        for field in data:
            #print(field)
            if field == "shapes":
                new_data[field] = data[field]
                new_data[field][shape] = data[field][shape]
                shp = data[field][shape]
                for subfield in data[field][shape]:
                    #print(subfield)
                    if subfield == "points":
                        new_data[field][shape][subfield] = dsp_pts
                        shp[subfield] = dsp_pts
                        #print(new_data[field][subfield])
                    else:
                        #print(data[field][subfield])
                        new_data[field][shape] = data[field][shape]
                        new_data[field][shape][subfield] = data[field][shape][subfield]
                        shp[subfield] = data[field][shape][subfield]
            elif field == "imagePath":
                f_data = name
                new_data[field] = f_data
            elif field =="imageData":
                encoded = base64.b64encode(open(path_to_new_img, "rb").read())
                f_data = ""
                new_data[field] = encoded.decode("utf-8") 
            elif field =="imageWidth":
                f_data = ""
                new_data[field] = new_img.shape[1]
            elif field =="imageHeight":
                f_data = ""
                new_data[field] = new_img.shape[0]
            else:
                f_data = data[field]
                new_data[field] = f_data
    
    json_name = name.replace(".jpg",".json")
    dst_label_path = 'labels'
    if not os.path.exists(os.path.join(AUG_PATH,dst_label_path,json_name)):
        json.dump(new_data,open(os.path.join(AUG_PATH,dst_label_path,json_name),"w"),indent=2,separators=(", ",": "),sort_keys=False)
    else:
        update_data = json.load(open(os.path.join(AUG_PATH,dst_label_path,json_name)))
        update_data["shapes"].append(shp)
        encoded = base64.b64encode(open(path_to_new_img, "rb").read())
        update_data["imageData"] = encoded.decode("utf-8") 
        json.dump(update_data,open(os.path.join(AUG_PATH,dst_label_path,json_name),"w"),indent=2,separators=(", ",": "),sort_keys=False)

#------------------------------------------------------------------------------
def pastePNG(bg, patch, pos = [0,0]):
    """
    Pastes a png image in a specific point of a background image
    """
    hp, wp, cp = patch.shape
    hb, wb, cb = bg.shape
    _ ,_ ,_ , mask = cv2.split(patch)
    #print(np.array(mask).shape)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(patch, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb,wb,cb), np.uint8)
    imgMaskFull[pos[1]:hp + pos[1], pos[0]:wp + pos[0],:] = imgRGB
    imgMaskFull2 = np.ones((hb,wb,cb), np.uint8) * 255
    maskBGRInv = cv2.bitwise_not(maskBGR)
    imgMaskFull2[pos[1]:hp + pos[1], pos[0]:wp + pos[0], :] = maskBGRInv

    # Make a copy to avoid overwritting
    bg_copy = bg.copy()
    bg_copy = cv2.bitwise_and(bg_copy, imgMaskFull2)
    bg_copy = cv2.bitwise_or(bg_copy, imgMaskFull)

    return bg_copy

#------------------------------------------------------------------------------
def pasteBinPNG(bg, patch, idx_obj, pos = [0,0]):
    # Dimensions of the background and patch to be inserted:
    hp, wp, dp = patch.shape
    hb, wb, db = bg.shape
    
    # Creating mask image with the size of the background to include the new patch:
    imgMaskFull = np.zeros((hb,wb,db), np.uint8)

    # Binarizing current patch:
    patchGray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)	
    _, patchBW = cv2.threshold(patchGray, 0, 1, cv2.THRESH_BINARY)
    patchBW *= idx_obj
    patchBGR = cv2.cvtColor(patchBW, cv2.COLOR_GRAY2BGR)
    
    # Inserting the patch in the empty background image:
    imgMaskFull[pos[1]:hp + pos[1], pos[0]:wp + pos[0],:] = patchBGR
    _, imgMaskFull_fs = cv2.threshold(imgMaskFull, 0, 255, cv2.THRESH_BINARY)

    # Make a copy to avoid overwritting:
    bg_copy = bg.copy()


    # Overlapping area per object in the image (excluding background, indexed as 0):
    object_areas = dict()
    for it_object in np.delete(np.unique(bg_copy), 0):
        area_existing_object = np.count_nonzero(bg_copy == it_object)/3
        overlapping_area = np.count_nonzero(cv2.bitwise_and((bg_copy == it_object).astype(np.uint8)*255, imgMaskFull_fs))/3
        object_areas[it_object] = 100 * overlapping_area / area_existing_object
        
    # Including the mask of the new object in the overall mask image:
    bg_copy = bg_copy + imgMaskFull
    _, bg_copy = cv2.threshold(bg_copy, idx_obj, idx_obj, cv2.THRESH_TRUNC)

    return bg_copy, object_areas




#------------------------------------------------------------------------------
def patch_img(bg_bin, background, patch, idx_obj):
    """
    Creates a synthetic using a background image an a png image as a patch
    """

    paste = False
    
    # Background dimesions:
    h, w, d = background.shape

    # Iterating through the number of attempts:
    for i in range(TRY):
        print("\t- Attempt #{}".format(i))

        # Processing patch from in each attempt:
        p = patch.copy()
        p_h, p_w, _ = p.shape # Patch
        

        # Random rotation:
        if ROT:             
            # random.seed(1)
            angle = randrange(360)
            print(f"\t\t - Applying rotation of {angle}")
            p = ndimage.rotate(p, angle)
        else:
            angle = 0

        # Norm Rotated mask:
        if SCALE:               
            while True:
                # random.seed(1)
                rs = 1
                if min(h,w)/max(p_h,p_w) < 1: rs = min(h,w)/max(p_h,p_w) ######
                rs *= float( randrange(25,50) ) / 100.0
                p = cv2.resize(p,(0,0),None,fx=rs,fy=rs,interpolation=cv2.INTER_AREA)
                p_h, p_w, _ = p.shape
                print(f"\t\t - Applying resize: {rs}")
                if p_h < h and p_w < w:
                    break
                    
        else:
            rs = 1.0

        # Traslation (X/Y):
        if TRANS:               
            p_h, p_w, _ = p.shape
            # random.seed(1)
            x = randrange(w - p_w)
            # random.seed(1)
            y = randrange(h - p_h)
            print(f"\t\t - Applying paste in location:{x,y}")

        else:
            x,y = 0,0

        # Checking pasting feasibility:
        bin_seg, object_areas = pasteBinPNG(bg_bin, p, idx_obj, [x,y])
        if checkOverlap(bg_bin,bin_seg, object_areas): 
            bg_bin = bin_seg
            paste = True
            break

    if paste: background = pastePNG(background, p, [x,y])

    return bg_bin, background, angle, x, y, rs, paste


#------------------------------------------------------------------------------
def obj_selector(bg_dataset,objects_dataset):
    objects, labels = obj_dataloader(objects_dataset)
    bgs = bg_dataloader(bg_dataset)

    # Adding elements in bgs until reaching original length:
    aux = sample(bgs, 990-len(bgs))
    bgs.extend(aux)
    
    n = 0
    # Hash table which each value has an array of pairs containing the objects/labels to be applied in each bg
    # {0 : [(obj1_img,lable1),(obj9_img,label9)], 1: [(obj2_img,label2),(obj50_img,label50),(obj5_img,label5)] 
    bg_objs = {}
    for bg in tqdm((bgs),"Choosing objects"):
        n_objs = randrange(1,10)
        objs = []
        for i in range(n_objs):
            # Retrieving object to introduce:
            index = randrange(0,len(objects))
            objs.append((objects[index],labels[index]))
        #Create hash table for every bg image and its objects associated
        bg_objs[n] = objs
        n+=1
    return bgs,bg_objs

#------------------------------------------------------------------------------
def create_new_dataset(bg_dataset, objects_dataset):
    bgs, bg_objs = obj_selector(bg_dataset, objects_dataset)

    # Iterate through bg images
    for bg_idx in tqdm((bg_objs), "Creating Dataset"):
        print("")
        bg = bgs[bg_idx]
        nimg_name = f"image_{bg_idx}" + ".jpg"

        # Init target mask:
        bin_bg = np.zeros(bg.shape, np.uint8)

        # Iterate through objects
        for idx_obj, objs in enumerate(bg_objs[bg_idx]):
            img_obj = objs[0]
            label = objs[1]

            print("- Pasting {} object".format(label.split("/")[2]))

            # Patch image pasting with N attempts::
            bin_bg, bg, rotation, x, y, rs, patched = patch_img(bg_bin = bin_bg, background = bg, patch = img_obj, idx_obj = idx_obj + 1)

            if patched: 
                # Image:
                cv2.imwrite(os.path.join(AUG_PATH, "images", nimg_name), bg)

                # Annotation:
                patch_json(nimg_name, label, rotation, x, y, rs, img_obj, bg, os.path.join(AUG_PATH, "images", nimg_name))
                

#------------------------------------------------------------------------------
def checkOverlap(bin_img, obj_bin_bg, object_areas):
    # overlap = bin_img + obj_bin_bg
    _, aux = cv2.threshold(obj_bin_bg, 0, 255, cv2.THRESH_BINARY)
    overlap = bin_img + aux
    cv2.imwrite("overlap.png", overlap)
    _, aux = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite("clean_bg.png", aux)
    _, aux = cv2.threshold(obj_bin_bg, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite("with_obj_bg.png", aux)

    
    if (np.array(list(object_areas.values())) > OVERLAP).any():
        print("\t --> Unsuccessful paste...")
        return False
    else:
        print("\t --> Paste Successful!!!")
        return True    

#------------------------------------------------------------------------------
def create_dataset(bg_dataset, objects_dataset):
    """
    Creates a synthetic dataset using background images and previously labeled objects
    """
    if not os.path.exists(os.path.join(AUG_PATH,"images")):
        os.makedirs(os.path.join(AUG_PATH,"images"))

    if not os.path.exists(os.path.join(AUG_PATH,"labels")):
        os.makedirs(os.path.join(AUG_PATH,"labels"))

    objects, labels = obj_dataloader(objects_dataset)
    bgs = bg_dataloader(bg_dataset)
    n = 0
    for bg in tqdm((bgs), "Creating Dataset"):
        n_objs = randrange(1,10)
        nimg_name = f"image_{n}" + ".jpg"
        for i in range(n_objs):
            # Retrieving object to introduce:
            index = randrange(0,len(objects))
            img_obj = objects[index]
            label = labels[index]

            # Introducing object in the image:
            bg,rotation,x,y,rs = patch_img(bg, img_obj)

            # Saving the image:
            cv2.imwrite(os.path.join(AUG_PATH,"images",nimg_name),bg)

            # Saving the json file:
            patch_json(nimg_name,label,rotation,x,y,rs,img_obj,bg,os.path.join(AUG_PATH,"images",nimg_name))
            
        n +=1

#------------------------------------------------------------------------------
def obj_dataloader(dataset):
    """
    Loads all objects and labels from the specified dataset
    """
    img_array, json_array = [],[]
    for dir in tqdm(os.listdir(dataset), "Cargando Objetos"):
        dir_path = os.path.join(dataset, dir)
        n=0

        #Para cada imagen en la carpeta de mascaras
        for file_object in tqdm(os.listdir(dir_path), f"Cargando objeto {dir}",leave= False):
            if file_object.endswith(".png"):
                img_array.append(cv2.imread(os.path.join(dir_path, file_object), cv2.IMREAD_UNCHANGED))
                json_path = os.path.join(dir_path,file_object)
                json_array.append(json_path.replace(".png",".json"))

    return img_array,json_array

#------------------------------------------------------------------------------
def bg_dataloader(dataset):
    """
    Loads all background images from the specified dataset
    """
    img_array = []

    #Para cada imagen en la carpeta de fondos
    for file_object in tqdm(os.listdir(dataset), f"Cargando images de fondo"):
        if file_object.endswith(".png"): 
            img_array.append(cv2.imread(os.path.join(dataset, file_object)))

    return img_array

#------------------------------------------------------------------------------
def checkMasks():
    for img_p in tqdm(os.listdir(AUG_PATH)):
        if img_p.endswith(".jpg"):
            image, mask = data_visualization.load_image_label(img_p, AUG_PATH)
            augmenter = data_visualization.obtain_augmenter(image.shape,mask)
            #print(image.shape)
            #print(mask.shape)
            images_aug, segmaps_aug = augmenter(image=image, segmentation_maps=SegmentationMapsOnImage(mask, shape=image.shape))
            #print("generate")
            #print(e_time-s_time)
            grid_image = data_visualization.draw_result(image, SegmentationMapsOnImage(mask, shape=image.shape), images_aug,segmaps_aug)
            if not os.path.exists(os.path.join(AUG_PATH,"Visualization")):
                os.makedirs(os.path.join(AUG_PATH,"Visualization"))
            imageio.imwrite(os.path.join(AUG_PATH,"Visualization",img_p.replace(".jpg",f"_mask.jpg")), grid_image)
            #cv2.imshow("Mask",grid_image)
            #cv2.waitKey(0)
#------------------------------------------------------------------------------

if __name__ == "__main__":
    MASK_DATASET = "synthetic_data/Objetos"
    BG_DATASET = "synthetic_data/Backgrounds"

    OVERLAP = 10 # int(input("Introduce percentage of overlapping allowed:\n"))
    TRY = 2 # int(input("Introduce number of attempts for pasting objects:\n"))

    dst_path = 'synthetic_data/images'
    files = [u for u in os.listdir(dst_path) if os.path.isfile(os.path.join(dst_path, u))]
    for u in files: os.remove(os.path.join(dst_path, u))

    dst_path = 'synthetic_data/labels'
    files = [u for u in os.listdir(dst_path) if os.path.isfile(os.path.join(dst_path, u))]
    for u in files: os.remove(os.path.join(dst_path, u))

    create_new_dataset(BG_DATASET, MASK_DATASET)
    #create_dataset(BG_DATASET, MASK_DATASET)
    checkMasks()

