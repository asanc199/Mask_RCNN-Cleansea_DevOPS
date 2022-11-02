from mrcnn.config import Config


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
    IMAGES_PER_GPU = 8

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Numero de clases + el background
    NUM_CLASSES = 1 + 19  # Cleansea tiene 19 clases

    # Salta las detecciones con <50% de seguridad
    DETECTION_MIN_CONFIDENCE = 0.5

    #Learning Rate Modificado
    LEARNING_RATE = 0.001



class InferenceConfig(CleanSeaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False



if __name__ == '__main__':
    config = CleanSeaConfig()
    config.display()
