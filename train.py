import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from imgaug import augmenters as iaa
import data_augmentation
import warnings
warnings.filterwarnings(action='ignore')


# Import mrcnn libraries
# sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib

# Import configuration:
from CleanSeaConfig import CleanSeaConfig, InferenceConfig
from CleanSeaDataset import CleanSeaDataset


# Import argument parsing:
import argument_parsing



# Dictionary from instance to material level:
mapping_dictionary = {
    # bottle, plastic_bag, glove, fishing_net, tire, plastic_debris, pipe -> plastic
    '4' : 1,
    '5' : 1,
    '6' : 1,
    '7' : 1,
    '8' : 1,
    '14' : 1,
    '16' : 1,
    # can, squared_can, wasingmachine, metal_chain, metal_debris, car_bumper -> metal
    '1' : 2,
    '2' : 2,
    '10' : 2,
    '11' : 2,
    '15' : 2,
    '18' : 2,
    # wood -> wood
    '3' : 3,
    # packaging_bag -> paper
    '9' : 4,
    # rope, towel, shoe, basket -> other
    '12' : 5,
    '13' : 5,
    '17' : 5,
    '19' : 5,
}



""" Train process """
def train_process(args):
    physical_devices = tf.config.list_physical_devices('GPU')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Directorio perteneciente a MASK-RCNN
    ROOT_DIR = './'
    MODEL_DIR = os.path.join(ROOT_DIR, "Models")

    # Creating path to models:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Path to weights file:
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Downloading pre-trained COCO weights:
    if not os.path.exists(COCO_WEIGHTS_PATH):
        utils.download_trained_weights(COCO_WEIGHTS_PATH)


    # Loading RCNN train configuration:
    config = CleanSeaConfig()
    config.display()

    """Train the model."""
    # Train partition:
    dataset_train = CleanSeaDataset()
    print("--- Train configuration ---")

    # Selecting either real or synthetic train data:
    if args.train_db == 'real':
        dataset_train.load_data("./CocoFormatDataset", "train_coco", size_perc = args.size_perc, filling_set = args.fill_db, limit_train = args.limit_train)
    else:
        dataset_train.load_data("./SynthSet", "train_coco", size_perc = args.size_perc)
    print("\t- Done loading data!")

    # Preparing data:
    dataset_train.prepare()
    print("\t- Done preparing train data!")

    # # Load and display random samples
    # print("Mostrando Imagenes aleatorias...\n")

    # image_ids = np.random.choice(dataset_train.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Instantiating a model (new):
    print("Initializing train model...\n")
    model = modellib.MaskRCNN(mode = "training", config = config, model_dir = MODEL_DIR)
    print("\t - Done!")

    # Init weights:
    if args.pretrain == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)

    elif args.pretrain == "coco":
        # Load weights trained on MS COCO, but skip layers that  are different due to the different number of classes See README for instructions to download the COCO weights
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # Selecting Data Augmentation type.
    seq = None
    if args.augmentation == 'mild':
        seq = data_augmentation.createMildDataAugmentation()
    elif args.augmentation == 'severe':
        seq = data_augmentation.createSevereDataAugmentation()


    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    print("Training Heads (first stage)...")
    model.train(dataset_train, learning_rate = config.LEARNING_RATE,\
        epochs = 5, layers = 'heads', augmentation = seq)
    print("\t - Done!")

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    for epoch_break_point in args.epochs:
        print("Training network (second stage)...")
        model.train(train_dataset = dataset_train, learning_rate = config.LEARNING_RATE / 10,\
            epochs = epoch_break_point, layers = "all", augmentation = seq)

        # Output name:
        MODEL_NAME = "Mask_RCNN_Epoch-{}_Aug-{}_Size-{}_Train-{}_Fill-{}_Limit-{}.h5".format(epoch_break_point,\
            args.augmentation, args.size_perc, args.train_db, args.fill_db, args.limit_train)

        # Save weights
        print("\t -Saving weights in {}...\n".format(os.path.join(MODEL_DIR, MODEL_NAME)))
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        model.keras_model.save_weights(model_path)
        print("\t - Done!")

    return


""" Inference process """
def inference_process(args):

    physical_devices = tf.config.list_physical_devices('GPU')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Test partition:
    dataset_test = CleanSeaDataset()
    print("\n--- Test configuration ---")
    if args.test_db == 'real':
        dataset_test.load_data("./CocoFormatDataset", "test_coco")
    else:
        dataset_test.load_data("./SynthSet", "test_coco")

    print("\t- Done loading data")
    dataset_test.prepare()
    print("\t- Done preparing test data!")

    # Loading inference configuration for the RCNN model:
    inference_config = InferenceConfig()

    # Recreate the model in inference mode:
    ROOT_DIR = './'
    MODEL_DIR = os.path.join(ROOT_DIR, "Models")

    # Creating the RCNN neural model:
    model = modellib.MaskRCNN(mode = "inference", config = inference_config, model_dir = MODEL_DIR)

    for epoch_break_point in args.epochs:
        # Retrieving model name:
        MODEL_NAME = "Mask_RCNN_Epoch-{}_Aug-{}_Size-{}_Train-{}_Fill-{}.h5".format(epoch_break_point, args.augmentation,\
            args.size_perc, args.train_db, args.fill_db)

        # Get path to saved weights
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)

        # Load trained weights
        print("Loading weights from {}...".format(model_path))
        model.load_weights(model_path, by_name=True)
        print("\t - Done!")

        # Iterating through the different test images:
        APs_025_instance = list()
        APs_05_instance = list()
        APs_075_instance = list()
        AP_ranges_instance = list()
        IoUs_instance = list()
        APs_05_material = list()
        APs_075_material = list()
        APs_025_material = list()
        IoUs_material = list()
        AP_ranges_material = list()
        for image_id in dataset_test.image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, inference_config, image_id)


            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            

            #### /INSTANCE-LEVEL EVALUATION\ ###
            # Compute AP
            # th = 0.5
            AP_05, precisions, recalls, overlap = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], 0.5)
            APs_05_instance.append(AP_05)

            # th = 0.75
            AP_075, precisions, recalls, overlap = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], 0.75)
            APs_075_instance.append(AP_075)

            # th = 0.25
            AP_025, precisions, recalls, overlap = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], 0.25)
            APs_025_instance.append(AP_025)

            # AVG Intersection over Union:
            IoUs_instance.append(np.average(overlap))

            # COCO range
            AP_range = utils.compute_ap_range(gt_box = gt_bbox, gt_class_id = gt_class_id, gt_mask = gt_mask,\
                     pred_box = r["rois"], pred_class_id = r["class_ids"], pred_score = r["scores"], pred_mask = r['masks'],\
                     iou_thresholds = None, verbose = 0)
            AP_ranges_instance.append(AP_range)
            #### \INSTANCE-LEVEL EVALUATION/ ###



            #### /MATERIAL-LEVEL EVALUATION\ ###
            # Grouping classes:
            gt_class_id_material = np.array([mapping_dictionary[str(u)] for u in gt_class_id], dtype=np.int32)
            class_ids_material = np.array([mapping_dictionary[str(u)] for u in r["class_ids"]],dtype=np.int32)

            # Compute AP
            # th = 0.5
            AP_05, precisions, recalls, overlap = utils.compute_ap(gt_bbox, gt_class_id_material, gt_mask, r["rois"], class_ids_material, r["scores"], r['masks'], 0.5)
            APs_05_material.append(AP_05)

            # th = 0.75
            AP_075, precisions, recalls, overlap = utils.compute_ap(gt_bbox, gt_class_id_material, gt_mask, r["rois"], class_ids_material, r["scores"], r['masks'], 0.75)
            APs_075_material.append(AP_075)

            # th = 0.25
            AP_025, precisions, recalls, overlap = utils.compute_ap(gt_bbox, gt_class_id_material, gt_mask, r["rois"], class_ids_material, r["scores"], r['masks'], 0.25)
            APs_025_material.append(AP_025)

            # AVG Intersection over Union:
            IoUs_material.append(np.average(overlap))

            # COCO range
            AP_range = utils.compute_ap_range(gt_box = gt_bbox, gt_class_id = gt_class_id_material, gt_mask = gt_mask,\
                     pred_box = r["rois"], pred_class_id = class_ids_material, pred_score = r["scores"], pred_mask = r['masks'],\
                     iou_thresholds = None, verbose = 0)
            AP_ranges_material.append(AP_range)
            #### \MATERIAL-LEVEL EVALUATION/ ###



        print("--- Results --- ")    
        print(" - Instance - ")
        print("\t - IoU ({}): {:.2f}".format(MODEL_NAME, np.nanmean(IoUs_instance)))
        print("\t - mAP-Range ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(AP_ranges_instance)))
        print("\t - mAP-0.25 ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(APs_025_instance)))
        print("\t - mAP-0.5 ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(APs_05_instance)))
        print("\t - mAP-0.75 ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(APs_075_instance)))

        print(" - Material - ")
        print("\t - IoU ({}): {:.2f}".format(MODEL_NAME, np.nanmean(IoUs_material)))
        print("\t - mAP-Range ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(AP_ranges_material)))
        print("\t - mAP-0.25 ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(APs_025_material)))
        print("\t - mAP-0.5 ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(APs_05_material)))
        print("\t - mAP-0.75 ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(APs_075_material)))



    return



if __name__ == '__main__':
    # Parameters:
    args = argument_parsing.menu()

    if args.process == 'train':
        # Performing the training:
        train_process(args)
    elif args.process == 'inference':
        # Inference stage:
        inference_process(args)
    else:
        # Performing both train and inference:
        ### Train phase:
        train_process(args)

        ### Inference phase:
        inference_process(args)
