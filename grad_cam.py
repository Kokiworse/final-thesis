from keras.applications import inception_v3
import numpy as np
import cv2
import keras.backend as K
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import time
from load_img import load
import sys
from guided_grad_cam import guided_back_prop_heatmap


def grad_cam_heatmap(original):
    model = inception_v3.InceptionV3(weights = 'imagenet', input_shape = None)
    processed_image, original = prep(original)
    predictions = model.predict(processed_image)
    
    label = decode_predictions(predictions)
    class_idx = np.argmax(predictions[0]) #topmost class index
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer('conv2d_94') #output for the last conv layer (from model.summary())

    # We compute the gradient of the class output value with respect to the feature map.
    # Then, we pool the gradients over all the axes leaving out the channel dimension.
    # Finally, we weigh the output feature map with the computed gradient values.
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([processed_image])
    for i in range(conv_layer_output_value.shape[2]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0) #relu
    heatmap /= np.max(heatmap)
    #show the original img and the cam
    #img = cv2.resize (img , (800, 600))
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original, 0.4, heatmap, 0.6, 0)
    return superimposed_img, heatmap

def prep(img):
    original = cv2.imread(img)
    original = original[ :, 0 : 2560//2] #crop in half
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = inception_v3.preprocess_input(image_batch.copy())
    return processed_image, original


if __name__ == "__main__":
    img = "football.jpg"
    hm = grad_cam_heatmap(img)

    # cv2.imshow("Original", img)
    dirName = "C:\\Users\\koki\\Desktop\\Conv94Grey"
    #
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    # print(t, "/", l , "image")
    # cv2.imwrite("C:\\Users\\koki\\Desktop\\Conv94Grey\\norm{}.jpg".format(t), original)
    cv2.imshow("CAM", hm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("C:\\Users\\koki\\Desktop\\Conv94Grey\\cam{}.jpg".format(t), superimposed_img)