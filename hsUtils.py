import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mrcnn import utils, visualize
import time

def get_ax(rows = 1, cols = 1, size = 16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def visualise_annotations(dataset, n_instances= None):
    if n_instances == None:
        size = len(dataset.image_ids)
    else:
        size = n_instances
    
    for image_id in range(size):
        image = dataset.load_image(image_id)
        if image.shape[-1] == 4: # PNG images have alpha channel as the 4th channel. 
            image = image[..., :3] # Drop the 4th channel
            
        mask, class_ids = dataset.load_mask(image_id)
        
        bbox = utils.extract_bboxes(mask)
        
        print("image_id ", image_id, dataset_t.image_reference(image_id))
        log("image", image)
        log("mask", mask)
        log("class_ids", class_ids)
        log("bbox", bbox)
                
        ax = get_ax(1)
        
        try:
            visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, ax=ax)
        except ValueError:
            print("Image size and Mask size does not match")
            
def visualise_annotation_by_pos(dataset, position):
    
    image = dataset.load_image(position)
    if image.shape[-1] == 4:
        image = image[..., :3]
        
    mask, class_ids = dataset.load_mask(position)
    
    bbox = utils.extract_bboxes(mask)
    
    print("image_id ", image_id, dataset_t.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
                
    ax = get_ax(1)
    
    try:
        visualize.display_instances(image, bbox, maks, class_ids, dataset.class_names, ax=ax)
    except ValueError:
        print("Image size and Mask size does not match")
        
        
def detectHS(directory, image_names, model, save=False):
    import os
    class_names = [
        'BG',
        'Tunnel Draining',
        'Scar HT',
        'Abcess',
        'NoName',
        'Papule Inf',
        'Komedo One',
        'Ulcer',
        'Tunnel NonDraining',
        'Plaque NonInf',
        'Scar Atrophic',
        'Komedo Two',
        'Plaque Inf',
        'Pustule',
        'Nodule Inf'
        ]
    
    for img_path in image_names:
        image_path = os.path.join(directory, img_path)
        img = mpimg,imread(image_path)
        
        result = model.detect([img], verbose = 1)
        ax = get_ax(1)
        r = result[0]
        visualize.display_instances(img, r['rois'], r['masks'],r['class_ids'],
                                    class_names,r1['scores'],ax = ax,
                                    title="Prediction")
        if save:
            
            save_base = "/content/drive/MyDrive/dataset/"
            timestr = time.strftime("%Y%m%dT%H%M%S")
            s_path = os.path.join(save_base,timestr)
            os.mkdir(s_path)
            
            save_path = img_path + "pred" + ".JPG"
            save_path = os.path.join(s_path, save_path)
            plt.savefig(save_path)