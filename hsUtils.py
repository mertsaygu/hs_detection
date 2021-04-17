import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mrcnn import utils, visualize
from mrcnn.model import log 
import time, os

def get_ax(rows = 1, cols = 1, size = 16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def visualise_annotations(dataset, n_instances= None):
    '''
    dataset : hsDataset
    n_instances : Number of images. Default is 'None'. When it is None takes entire dataset
                n_instances must be in between 1 and the size of the dataset. 
    '''
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
        
        print("image_id ", image_id, dataset.image_reference(image_id))
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
    '''
    dataset : hsDataset
    position : Index of the desired image
    '''
    image = dataset.load_image(position)
    if image.shape[-1] == 4:
        image = image[..., :3]
        
    mask, class_ids = dataset.load_mask(position)
    
    bbox = utils.extract_bboxes(mask)
    
    print("image_id ", position, dataset.image_reference(position))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
                
    ax = get_ax(1)
    
    try:
        visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, ax=ax)
    except ValueError:
        print("Image size and Mask size does not match")
               
def detectHS(model, directory, image_names, save=False, save_dir = None, name_tag = None):
    '''
    model : Mask rcnn model
    directory : Folder that includes images to detect tissues on.
    image_names : List of image names that are in the directory example : ["1.JPG","2.JPG]...
    save : Save option. Default 'False'
    save_dir : Directory to save annotated images
    '''
    
    
    if save:
        assert save_dir != None, "Save directory must be string, not 'None'"
        assert os.path.exists(save_dir),f"'{save_dir}' does not exist in your file system! Please make sure that the save directory exists"
    
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
        assert img_path in os.listdir(directory),f"'{img_path}' is not in the given directory"
        image_path = os.path.join(directory, img_path)
        img = mpimg.imread(image_path)
        
        result = model.detect([img], verbose = 1)
        ax = get_ax(1)
        r = result[0]
        visualize.display_instances(img, r['rois'], r['masks'],r['class_ids'],
                                    class_names,r['scores'],ax = ax,
                                    title="Prediction")
        if save:
            timestr = time.strftime("%Y%m%dT%H%M%S")
            
            save_path = img_path.split(".")[0] + name_tag + "_pred_"+timestr + ".JPG"
            save_path = os.path.join(save_dir, save_path)
            plt.savefig(save_path)
            
def detectHS_from_hsdataset(model, dataset, n_instances = None, save = False, save_dir = None):
    '''
    model : Mask rcnn model
    dataset : hsDataset object 
    n_instances : the number of images that you want to detect hs tissue. Default is 'None' which means entire dataset.
    save : Save option. Default 'False'. İf 'True' saves detection images to given path.
    save_dir : If option save is 'True', Model saves the detection image to the given path. If given folder does not exist, raises an AssertionError 
    '''
    max_size = dataset.image_ids
    if save:
        assert save_dir != None, "Path cannot be 'None'. Please pass a valid save directory!"
        assert os.path.exists(save_dir), f"'{save_dir}' does not exist in your file system! Please make sure that the save directory exists"
        
    if n_instances == None:
        n_instances = max_size
    else:
        assert max_size >= n_instances, f"Dataset does not include {n_instances} images. You can try something in between [1-{max_size}]." 
        assert n_instances != 0, "The number of instances can not be 0."
    
    for img_id in n_instances:
        image = dataset.load_image(img_id)
        
        result = model.detect([image], verbose = 1)
        ax = get_ax(1)
        r = result[0]
        visualize.display_instances(image, r["rois"],r["masks"],r["class_ids"],
                                    dataset.class_names,r["scores"], ax=ax,
                                    title= "Prediction "+str(img_id+1))
        if save:
            timestr = time.strftime("%Y%m%dT%H%M%S")
            save_name = str(img_id+1)+"_pred_"+timestr+".JPG"
            save_path = os.path.join(save_dir,save_name)
            plt.savefig(save_path)