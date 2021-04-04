from mrcnn import utils
import os
import skimage.draw, skimage.io
import numpy as np 

class hsDataset(utils.Dataset):
    def load_hsdata(self,dataset_dir, subset):
        self.add_class("HS", 1, "Püstül")
        self.add_class("HS", 2, "Komedon tekli")
        self.add_class("HS", 3, "Komedon ikili")
        self.add_class("HS", 4, "Nodül inflamatuvar")
        self.add_class("HS", 5, "Apse")
        self.add_class("HS", 6, "Nodül")
        self.add_class("HS", 7, "Ülser")
        self.add_class("HS", 8, "Skar hipertrofik")
        self.add_class("HS", 9, "Fistül direne")
        self.add_class("HS", 10, "Fistül")
        self.add_class("HS", 11, "Skar atrofik")
        self.add_class("HS", 12, "Komedon üçlü")
        self.add_class("HS", 13, "Piyojenik granülozum")
        
        assert subset in ["train","val"]
        dataset_dir = os.path.join(dataset_dir,subset)
        
        _annotations = json.load(open(os.path.join(dataset_dir, "via_project.json")))
        annotations = list(_annotations.values())
        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['HS'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"Püstül":1, "Komedon tekli":2,                                     
                      "Komedon ikili":3,                                    
                      "Nodül inflamatuvar":4,                               
                      "Apse": 5,                                              
                      "Nodül" : 6,                                            
                      "Ülser": 7,                                             
                      "Skar hipertrofik" : 8,                                 
                      "Fistül direne" : 9,                                     
                      "Fistül" : 10,                                           
                      "Skar atrofik" : 11,                                     
                      "Komedon üçlü" : 12,                                    
                      "Piyojenik granülozum" : 13 
                    }
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
     
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            # image = skimage.io.imread(image_path)
            image = skimage.io.imread(image_path, plugin='pil')
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
            
            image_info = self.image_info[image_id]
            
            if image_info["source"] != "HS":
                return super(self.__class__,self).load_mask(image_id)
            
            info = self.image_info[image_id]
        if info["source"] != "HS":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # the dataset we are using has 3 types of annotation type -> Polygon, Circle, Ellipse
            # In order to load mask we use skimage 
                if p['name'] == 'polygon':
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
            
    