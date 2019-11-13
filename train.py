from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


''' Script to re-train the new model and perform transfer learning
'''

#load with no top for transfer learning
base_model = InceptionV3(weights='imagenet', include_top=False)

#training part (it shoudnt require lot of epochs)
EPOCHS = 15
BATCH_SIZE = 128
STEPS_PER_EPOCH = 64
VALIDATION_STEPS = 32
IMG_WIDTH, IMG_HEIGHT = 200, 200
TRAIN_DATA_DIR = 'data/train'
TEST_DATA_DIR = 'data/test'
VERBOSE = True
MODEL_FILE = 'First_model.h5'

#add a GAP layer and the layer for outs
CLASSES = 7 #n of new classes
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)
predictions = Dense(CLASSES, activation='softmax', name='output_dense')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#freeze the weights so you train only the new ones
for layer in base_model.layers:
    layer.trainable = False

#compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.applications.inception_v3 import preprocess_input
#create a generator for the training to augment the dataset (params to be checked)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function = preprocess_input,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode='nearest')

#create the test split
test_datagen = ImageDataGenerator(rescale=1. / 255)

#read from the dir for the training set(generator)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical')

#read from the dir for the test set(generator)
validation_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    verbose=VERBOSE,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)

model.save(MODEL_FILE)
