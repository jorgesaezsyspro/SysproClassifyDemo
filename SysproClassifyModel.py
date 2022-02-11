from keras import Sequential, layers
import tensorflow as tf
from tensorflow import losses
import tensorflow_addons as tfa
import numpy as np

# Define a simple sequential model

# TODO implement https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/data_augmentation.ipynb#scrollTo=r1Bt7w5VhVDY
# data augmentation outside of the main network

# data_augmentation = Sequential([
#         layers.RandomFlip(mode='horizontal_and_vertical', input_shape=(img_height, img_width, 3)),
#         layers.RandomRotation(0.2),
#         layers.RandomZoom(0.2),
#         layers.Resizing(img_height, img_width)
#     ])

# Funcion de lectura de imagenes y aumento
def augment_image(img, img_height, img_width, rotate=0, Hflip=False, Vflip=False, brightness=0, zoom=0, contrast=0):
    # adapatation from https://stackoverflow.com/questions/65475057/keras-data-augmentation-pipeline-for-image-segmentation-dataset-image-and-mask

    # zoom in a bit
    if zoom != 0 and (tf.random.uniform(()) > 0.5):
        # use original image to preserve high resolution
        img = tf.image.central_crop(img, zoom)
        # resize
        img = tf.image.resize(img, [img_height, img_width])
    
    # random brightness adjustment illumination
    if brightness != 0:
        img = tf.image.random_brightness(img, brightness)
    # random contrast adjustment
    if contrast != 0:
        img = tf.image.random_contrast(img, 1-contrast, 1+2*contrast)
    
    # flipping random horizontal 
    if tf.random.uniform(()) > 0.5 and Hflip:
        img = tf.image.flip_left_right(img)
    # or vertical
    if tf.random.uniform(()) > 0.5 and Vflip:
        img = tf.image.flip_up_down(img)

    # rotation in 360° steps
    if rotate != 0:
        rot_factor = tf.cast(tf.random.uniform(shape=[], minval=-rotate, maxval=rotate, dtype=tf.int32), tf.float32)
        angle = np.pi/360*rot_factor
        img = tfa.image.rotate(img, angle)

    return img


def classify_model(img_height, img_width, num_classes):

    model = Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu',
                      input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(),
        layers.Dropout(0.15),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.15),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.15),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model
