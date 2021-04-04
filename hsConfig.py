from mrcnn import Config

class hsConfig(Config):
    NAME="HS"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 14 # 13 different tissue and 1 background
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.8