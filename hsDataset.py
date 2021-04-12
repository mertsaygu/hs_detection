from mrcnn import utils
import os
import skimage.draw, skimage.io
import numpy as np 
import json

class hsDataset(utils.Dataset):
    
    def load_hsdata(self,dataset_dir, subset):
       
        self.add_class("HS", 1, 'Tunnel Draining')
        self.add_class("HS", 2, 'Scar HT')
        self.add_class("HS", 3, 'Abcess')
        self.add_class("HS", 4, 'NoName')
        self.add_class("HS", 5, 'Papule Inf')
        self.add_class("HS", 6, 'Komedo One')
        self.add_class("HS", 7, 'Ulcer')
        self.add_class("HS", 8, 'Tunnel NonDraining')
        self.add_class("HS", 9, 'Plaque NonInf')
        self.add_class("HS", 10, 'Scar Atrophic')
        self.add_class("HS", 11, 'Komedo Two')
        self.add_class("HS", 12, 'Plaque Inf')
        self.add_class("HS", 13, 'Pustule')
        self.add_class("HS", 14, 'Nodule Inf')
        
        assert subset in ["train","val"]
        dataset_dir = os.path.join(dataset_dir,subset)
        
        _annotations = json.load(open(os.path.join(dataset_dir, "via_project.json")))
        annotations = list(_annotations.values())
        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
            
            objects = [s['region_attributes']['HS'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {
                'Tunnel Draining': 1, 
                'Scar HT': 2, 
                'Abcess': 3, 
                'NoName': 4, 
                'Papule Inf': 5, 
                'Komedo One': 6, 
                'Ulcer': 7, 
                'Tunnel NonDraining': 8, 
                'Plaque NonInf': 9, 
                'Scar Atrophic': 10, 
                'Komedo Two': 11, 
                'Plaque Inf': 12, 
                'Pustule': 13, 
                'Nodule Inf': 14
                }
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
     
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path) # plugin = 'pil' rotates the vertical images to horizontal images which makes masks unusable 
            height, width = image.shape[:2]

            self.add_image(
                "HS",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )
        
    def load_mask(self, image_id):
            
        info = self.image_info[image_id]
        if info["source"] != "HS":
            return super(self.__class__, self).load_mask(image_id)
        
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # the dataset we are using has 3 types of annotation type -> Polygon, Circle, Ellipse
            # In order to load mask we use skimage 
                if p['name'] == 'polygon' or p['name'] == "polyline":
                    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])            
                elif p['name'] == 'circle':
                    rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
                else:  # Ellipse
                    rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'], rotation=np.deg2rad(p['theta']))  
                
                # Some labels may go outside of the image 
                # If there is such label exists we need to fix it
                # Following lines of codes will crop the labels.  
                # If dataset is correctly labeled You should not consider these
                rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                cc[cc > mask.shape[1]-1] = mask.shape[1]-1
                rr[rr < 0] = 0
                cc[cc < 0] = 0
                mask[rr, cc, i] = 1
        
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "HS":
            return info["path"]
        else:
            super(self.__class__,self).image_reference(image_id)
            
    