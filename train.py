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


def process(args):
    physical_devices = tf.config.list_physical_devices('GPU')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Directorio perteneciente a MASK-RCNN
    ROOT_DIR = './'
    MODEL_DIR = os.path.join(ROOT_DIR, "Models")

    # Descargamos los Pesos Entrenados de COCO
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Ruta al archivo de pesos
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Descargamos los Pesos Entrenados de COCO
    if not os.path.exists(COCO_WEIGHTS_PATH):
        utils.download_trained_weights(COCO_WEIGHTS_PATH)

    config = CleanSeaConfig()
    config.display()

    """Train the model."""
    # Training dataset.
    dataset_train = CleanSeaDataset()
    print("Configuracion para train cargada\n")
    dataset_train.load_data("./CocoFormatDataset", "train_coco", size_perc = args.size_perc)
    print("Dataset Inicializado Correctamente\n")
    dataset_train.prepare()
    print("Preparacion del Dataset Completada\n")
    
    print(dataset_train.num_classes)

    # return

    # Validation dataset
    dataset_test = CleanSeaDataset()
    print("Configuracion para test cargada\n")
    dataset_test.load_data("./CocoFormatDataset", "test_coco")
    print("Dataset Inicializado Correctamente\n")
    dataset_test.prepare()
    print("Preparacion del Dataset Completada\n")

    # Load and display random samples
    print("Mostrando Imagenes aleatorias...\n")

    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    print("Inicializing model for training...\n")
    model = modellib.MaskRCNN(mode = "training", config = config, model_dir = MODEL_DIR)

    # Which weights to start with?    
    if args.pretrain == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)

    elif args.pretrain == "coco":
        # Load weights trained on MS COCO, but skip layers that  are different due to the different number of classes See README for instructions to download the COCO weights
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # elif args.pretrain == "last":
    #     # Load the last model you trained and continue training
    #     model.load_weights(model.find_last(), by_name=True)

    #     last_path="./logs/mask_rcnn_debris_weights1000DA5Heads.h5"
    #     model.load_weights(last_path, by_name=True)


    ############################################################
    #  Training
    ############################################################
    seq = None
    if args.augmentation == 'mild':
        seq = data_augmentation.createMildDataAugmentation()
    elif args.augmentation == 'severe':
        seq = data_augmentation.createSevereDataAugmentation()


    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.

    print("Training Heads (first stage)...\n")
    model.train(dataset_train, dataset_test, learning_rate = config.LEARNING_RATE, epochs = 5, layers = 'heads', augmentation = seq)

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    for epoch_break_point in args.epochs:
        print("Training network (second stage)...\n")
        model.train(dataset_train, dataset_test, learning_rate = config.LEARNING_RATE / 10, epochs = epoch_break_point, layers = "all", augmentation = seq)

        # Output name:
        MODEL_NAME = "Mask_RCNN_Epoch-{}_Aug-{}_Size-{}.h5".format(epoch_break_point, args.augmentation, args.size_perc)

        # Save weights
        print("Saving weights in {}...\n".format(os.path.join(MODEL_DIR, MODEL_NAME)))
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        model.keras_model.save_weights(model_path)
        print("Done!")

    ############################################################
    #  Evaluacion
    ############################################################
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode = "inference", config = inference_config, model_dir = MODEL_DIR)

    for epoch_break_point in args.epochs:
        # Retrieving model name:
        MODEL_NAME = "Mask_RCNN_Epoch-{}_Aug-{}_Size-{}.h5".format(epoch_break_point, args.augmentation, args.size_perc)

        # Get path to saved weights
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)

        # Load trained weights
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        image_ids = dataset_test.image_ids
        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, inference_config, image_id)
            molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            
            # Compute AP
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
            
        print("mAP ({}): {:.2f}%".format(MODEL_NAME, 100*np.mean(APs)))

if __name__ == '__main__':
    # Parameters:
    args = argument_parsing.menu()

    # Performing the training:
    process(args)

    print("hello")