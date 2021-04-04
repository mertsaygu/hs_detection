from mrcnn import model as modellib
import hsDataset

config = hsConfig()
COCO_WEIGHTS_PATH = "/path/to/mask_rcnn_coco.h5"
DEFAULT_LOGS_DIR = "/logs_aug_ckp/"


def train(model):
    dataset_train = hsDataset()
    dataset_train.load_hsdata(args.dataset, "train")
    dataset_train.prepare()
    
    dataset_val = hsDataset()
    dataset_val.load_hsdata(args.dataset, "val")
    dataset_val.prepare()
    
    if args.augmentation:
        augmentation = imgaug.augmenters.Sometimes(0.2, [
                        imgaug.augmenters.Fliplr(0.5),
                        imgaug.augmenters.GaussianBlur(sigma=(0.0, 0.5)),
                        imgaug.augmenters.Affine(
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                            rotate=(-25, 25),
                            shear=(-8, 8)
                        )
                    ])
    else:
        augmentation = None
    
    model.train(dataset_train, dataset_val,
                learning_rate = config.LEARNING_RATE,
                epochs = 60,
                augmentation = augmentation,
                layers = "heads")