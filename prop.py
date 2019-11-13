from keras.applications import inception_v3
import numpy as np
import vis
import cv2
from vis.visualization import visualize_saliency
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions


def prep(img):
    original = cv2.imread(img)
    original = original[ :, 0 : 2560//2] #crop in half
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = inception_v3.preprocess_input(image_batch.copy())
    return processed_image, original


def get_layer_index(layer_name, model):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            index = idx
            break
    print(index)
    return index


def guided_back_prop_heatmap(img, model):
    img = cv2.imread(img)
    img = img[ : , 0 : 2560//2] #crop in half (only for my dataset)
    heatmap = visualize_saliency(model=model,
                                layer_idx= 299, # 299 getIndex("conv2d_94", model),
                                filter_indices=None,
                                seed_input=img,
                                backprop_modifier='guided',
                                grad_modifier='relu')

    heatmap = np.maximum(heatmap, 0) #relu
    heatmap /= np.max(heatmap)
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    print(heatmap.shape)
    return heatmap


def show_superimposed(img):
    cv2.imshow('Guided Backprop', hm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

