from keras import Sequential, layers
import tensorflow as tf
from tensorflow import losses
import tensorflow_addons as tfa
import numpy as np

AUTO = tf.data.experimental.AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE

# Define a simple sequential model

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

def resize_and_rescale(img, img_height=128, img_width=128): 
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, dtype=tf.float32) / 255.

    return img

# Funcion de lectura de imagenes y aumento
def augment_image(img, img_height=128, img_width=128, rotate=0, Hflip=False, Vflip=False, brightness=0, zoom=0, contrast=0):
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

    img = tf.clip_by_value(img, 0.0, 1.0)

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

def prepare(ds, augment=False, augment_param={}):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize_and_rescale(x, augment_param["img_height"], augment_param["img_width"]), y), 
              num_parallel_calls=AUTOTUNE)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (augment_image(x, **augment_param), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)