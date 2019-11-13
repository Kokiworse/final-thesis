import numpy as np
from prop import guided_back_prop_heatmap
from grad_cam import grad_cam_heatmap
import cv2
from load_img import load_images
from keras.applications import inception_v3


if __name__ == "__main__":
    #call script to load images
    imgs = load_images()

    #load model
    model = inception_v3.InceptionV3(weights = 'imagenet', input_shape = None)
    
    #loop through images
    for i, img in enumerate(imgs):
        heatmap = np.zeros((cam.shape[0], cam.shape[1], cam.shape[2]))
        
        #get grad-cam and back-prop heatmaps
        cam, cam1 = grad_cam_heatmap(img)
        guided_prop = guided_back_prop_heatmap(img, model)
        
        #normalization
        a = np.array(guided_prop / 255)
        b = np.array(cam1 / 255)
        
        #compute guided-grad-cam
        heatmap = np.multiply(a , b)
        heatmap = np.uint8(255 * heatmap)
        
        print(cam.shape, guided_prop.shape, heatmap.shape)
        print("%d/10"  %(i + 1))
        
        #save to file all superimposed images
        path = ("C:\\Users\\koki\\Desktop\\Report\\provaGcam %d.jpg" %i)
        cv2.imwrite(path, heatmap)
        path = ("C:\\Users\\koki\\Desktop\\Report\\provaCam %d.jpg" %i)
        cv2.imwrite(path, cam)
        path = ("C:\\Users\\koki\\Desktop\\Report\\provaBack %d.jpg" %i)
        cv2.imwrite(path,  guided_prop)

