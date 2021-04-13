from mrcnn.config import Config
class hsConfig(Config):
    NAME="HS"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 14 # 14 different tissue and 1 background
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.8
    
config = hsConfig()
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8