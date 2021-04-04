import skimage.draw, skimage.io
import datetime

def color_splash(image, mask):
    gray = skimage.draw.color.gray2rgb(skimage.color.rgb2gray(image))*255
    mask = (np.sum(mask, -1, keepdims = True) >= 1)
    
    if mask.shape[0]> 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash

def detect_and_color_splash(model, image_path= None, video_path = None):
    
    assert image_path or video_path
    
    if image_path:
        print(f"Running on {args.image}")
        
        image = skimage.io.imread(args.image)
        r = model.detect([image], verbose = 1)[0]
        splash = color_splash(image, r["masks"])
        file_name = "splash_{:%Y%m%dT%H%M%S.png}".format(datetime.datetime.now())
        skimage.io.imsave(file_name,splash)
    
    elif video_path:
        
        import cv2
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name, 
                                  cv2.VideoWriter_fourcc(*"MJPG"),
                                  fps, (width, height))
        
        count = 0
        success = True
        
        while success:
            print("frame ", count)
            success,image  = vcapture.read()
            if success :
                image = image[...,::-1]
                r = model.detect([image],verbose = 0)[0]
                splash = color_splash(image, r["masks"])
                splash = splash[...,::-1]
                vwriter.write(splash)
                count += 1
        vwirter.release()
    print("Video saved as ", file_name)