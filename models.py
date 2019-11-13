from keras.applications import VGG16, ResNet50, InceptionV3, VGG19, MobileNet, MobileNetV2, DenseNet121, NASNetLarge
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
from PIL import Image


''' Script to try different models
'''


def try_models():
    list = (VGG16, ResNet50, InceptionV3, VGG19, MobileNet, MobileNetV2, DenseNet121, NASNetLarge)
    model1 = list[0](weights='imagenet')
    model2 = list[1](weights='imagenet')
    model3 = list[2](weights='imagenet')
    model4 = list[3](weights='imagenet')
    model5 = list[4](weights='imagenet')
    model6 = list[5](weights='imagenet')
    model7 = list[6](weights='imagenet')
    model8 = list[7](weights='imagenet')

    img_list = ('img2.jpg', 'img3.jpg')
    for i in img_list:
        img = image.load_img(i, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds1 = model1.predict(x)
        preds2 = model2.predict(x)
        preds3 = model3.predict(x)
        preds4 = model4.predict(x)
        preds5 = model5.predict(x)
        preds6 = model6.predict(x)
        preds7 = model7.predict(x)
        img = image.load_img(i, target_size=(331,331))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds8 = model8.predict(x)

        print('1 Predicted: ', (preds1.shape))
        print('2 Predicted: ', decode_predictions(preds2, top=3)[0])
        print('3 Predicted: ', decode_predictions(preds3, top=3)[0])
        print('4 Predicted: ', decode_predictions(preds4, top=3)[0])
        print('5 Predicted: ', decode_predictions(preds5, top=3)[0])
        print('6 Predicted: ', decode_predictions(preds6, top=3)[0])
        print('7 Predicted: ', decode_predictions(preds7, top=3)[0])
        print('8 Predicted: ', decode_predictions(preds8, top=3)[0])


if __name__ == "__main__":
    try_models()