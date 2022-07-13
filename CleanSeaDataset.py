import os
import json
import numpy as np
from mrcnn import utils
from PIL import Image, ImageDraw

############################################################
#  Dataset
############################################################
class CleanSeaDataset(utils.Dataset):
    def load_data(self, dataset_dir, subset, size_perc = 100):
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

        print("Lengths: categories : {} - annotations : {} - images : {}".format(len(coco_json['categories']), len(coco_json['annotations']), len(coco_json['images'])))


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
        num_images = int(size_perc*len(coco_json['images'])/100)
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



if __name__ == '__main__':
    dataset_train = CleanSeaDataset()
    dataset_train.load_data("./CocoFormatDataset","train_coco")