import os
import json
import numpy as np
from mrcnn import utils
from PIL import Image, ImageDraw

############################################################
#  Dataset
############################################################
class CleanSeaDataset(utils.Dataset):
    def load_data(self, dataset_dir, subset, size_perc = 100, filling_set = 'none'):
        # Train or test partition:
        assert subset in ["train_coco", "test_coco"]

        # Path to the dataset:
        dataset_dir = os.path.join(dataset_dir, subset)

        # Loading the JSON file:
        with open(os.path.join(dataset_dir,"annotations.json")) as json_file:
            coco_json = json.load(json_file)
        print("\t - Done loading annotations | Summary:")

        print("\t\t - Categories : {}\n\t\t - Annotations : {}\n\t\t - Images : {}".format(len(coco_json['categories']),\
            len(coco_json['annotations']),\
            len(coco_json['images'])))

        # Loading class labels (utils.Dataset):
        print("\t - Adding labels to the dataset")
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                # print('Error: Class id for "{}" reserved for the background'.format(class_name))
                pass
            else:
                self.add_class(source_name, class_id, class_name)
        print("\t\t - Done!")

        # Storing the annotations:
        print("\t - Storing annotations")
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        print("\t\t- Done!")

        # Setting the number of images to read:
        num_images = int(size_perc*len(coco_json['images'])/100)
        print("\t - Loading {} images out of the available {}".format(num_images, len(coco_json['images'])))

        # Storing the computed number of images:
        seen_images = {}
        for image in coco_json['images'][:num_images]:
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
                
                # Path to the image:
                image_path = os.path.join(dataset_dir, image_file_name)

                # Image annotation:
                image_annotations = annotations[image_id]
                
                # Adding image using the corresponding method (utils.Dataset):
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
        print("\t\t - Done!")


        # Including images from another corpus (synthetic, in our case) in addition to the real ones:
        if filling_set != 'none':
            # Maximum possible number of real images: 
            max_number_images = len(coco_json['images'])

            # Setting the number of additional images to use (gap between actual real ones and maximum real ones):
            num_images = max_number_images - len(self.image_info)

            print("\t - Including {} synthetic images".format(num_images))

            # Path to filling dataset:
            filling_set_name = 'SynthSet'
            filling_dataset_dir = os.path.join(filling_set_name, 'train_coco')

            # Loading the JSON file:
            print("\t\t - Loading annotations")
            with open(os.path.join(filling_dataset_dir,"annotations.json")) as json_file:
                coco_json = json.load(json_file)
            print("\t\t\t - Done!")

            # Storing the annotations:
            annotations = {}
            print("\t\t - Storing annotations")
            for annotation in coco_json['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations:
                    annotations[image_id] = []
                annotations[image_id].append(annotation)
            print("\t\t\t - Done!")

            # Loading each image:
            print("\t\t - Loading synthetic images")
            for image in coco_json['images'][:num_images]:
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
                    
                    # Path to the selected image:
                    image_path = os.path.join(dataset_dir, image_file_name)

                    # Annotation of the image:
                    image_annotations = annotations[image_id]
                    
                    # Adding the new image:
                    self.add_image(
                        source=source_name,
                        image_id=image_id,
                        path=image_path,
                        width=image_width,
                        height=image_height,
                        annotations=image_annotations
                    )
            print("\t\t\t - Done!")

        return

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



if __name__ == '__main__':
    dataset_train = CleanSeaDataset()
    dataset_train.load_data("./CocoFormatDataset", "train_coco", size_perc = 50)

    print("Hello")
