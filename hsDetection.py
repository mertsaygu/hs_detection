from mrcnn import model as modellib
from hsDataset import hsDataset
from hsConfig import hsConfig
import os
from splashEffect import color_splash, detect_and_color_splash

ROOT = os.getcwd()
cfg = hsConfig()
COCO_WEIGHTS_PATH = "/path/to/mask_rcnn_coco.h5"
DEFAULT_LOGS_DIR = os.path.join(ROOT, "logs")



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
                learning_rate = cfg.LEARNING_RATE,
                epochs = 60,
                augmentation = augmentation,
                layers = "heads")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HS Detection"
    )
    parser.add_argument("command",
                        metavar = "<command>",
                        help="\'train\' or \'splash\'")
    
    parser.add_argument("--dataset",
                        required = False,
                        metavar="/path/to/dataset/",
                        help="Directory of HS dataset")
    
    parser.add_argument("--weights",
                        required= True,
                        metavar="/path/to/weights.h5",
                        help="Directory of COCO weights or Trained weights")
    
    parser.add_argument("--logs",
                        required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="path/to/logs/directory/",
                        help="Logs and checkpoints directory. Default /path/to/hsDetection.py/logs/")
    
    parser.add_argument("--image",
                        required=False,
                        metavar="Path or URL to image",
                        help=" Image to apply the color splash on")
    
    parser.add_argument("--video",
                        required= False,
                        metavar="Path or URL to video",
                        help="Video to applt the color splash effect on")
    
    args = parser.parse_args()
    
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, "Provide --image or --video to apply splash effect"
        
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    
     