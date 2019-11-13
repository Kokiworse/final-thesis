from keras.applications import inception_v3
import numpy as np
import cv2
import keras.backend as K
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import sys


''' First script trying to use inceptionv3 and using grad-cam method visualization
'''


imgs = ['football.jpg', 'bird.jpg', 'frog.jpg', 'car.jpg']
model = inception_v3.InceptionV3(weights = 'imagenet', input_shape = None)
t = 1


for img in imgs:

    original = cv2.imread(img)
    if original is None :
        break
    #original = original[ :, 0 : 2560//2] #crop in half
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = inception_v3.preprocess_input(image_batch.copy())

    predictions = model.predict(processed_image)

    label = decode_predictions(predictions)
    print(label)

    # model.summary()
    #i can get a map for every class
    #map for top prediction
    class_idx = np.argmax(predictions[0]) #topmost class index
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer('conv2d_94') #output for the last conv layer (from model.summary())
    #
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
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #show the original img and the cam
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    
    dir_name = "C:\\Users\\koki\\Desktop\\example2"
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    cv2.imwrite("C:\\Users\\koki\\Desktop\\example2\\norm{t}.jpg".format(t = t), original)
    cv2.imwrite("C:\\Users\\koki\\Desktop\\example2\\cam{t}.jpg".format(t = t), superimposed_img)
    t += 1
    # sys.exit(0)


sys.exit(0)
