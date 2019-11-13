from keras.applications import xception
import numpy as np
import cv2
import keras.backend as K
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions


''' First try to load inceptionv model and compute heatmap
'''


#299x299 input
model = xception.Xception(include_top=True, weights='imagenet', input_shape=None)


img = "prova3.jpg"
original = cv2.imread(img)

original = original[ :, 0 : 2560//2] #crop in half

original = cv2.resize (original , (299, 299)) #input size

numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)

processed_image = xception.preprocess_input(image_batch.copy())

predictions = model.predict(processed_image)
label = decode_predictions(predictions)
print(label)

# model.summary()
# conv2d_4 745472

class_idx = np.argmax(predictions[0]) #topmost class index
class_output = model.output[:, class_idx]
last_conv_layer = model.get_layer('conv2d_4') #output for the last conv layer (from model.summary())

#
# We compute the gradient of the class output value with respect to the feature map.
# Then, we pool the gradients over all the axes leaving out the channel dimension.
# Finally, we weigh the output feature map with the computed gradient values.


grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([processed_image])
for i in range(1024):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]


heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

#show the original img and the cam
img = "prova3.jpg"
img = cv2.imread(img)
img = img[ :, 0 : 2560//2] #crop in half
#img = cv2.resize (img , (800, 600))
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
cv2.imshow("Original", img)
cv2.imshow("Cam", superimposed_img)
key=cv2.waitKey(0)
if (key==27):
    cv2.destroyAllWindows()
