from keras import Sequential, layers, optimizers
import tensorflow as tf
from tensorflow import losses
import tensorflow_addons as tfa
import numpy as np
import random
import keras_tuner as kt
import matplotlib.pyplot as plt

AUTO = tf.data.experimental.AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE

class MyHyperModel(kt.HyperModel):
    def __init__(self, img_height, img_width, num_classes, name=None, tunable=True):
        super().__init__(name, tunable)
        self.img_height=img_height
        self.img_width=img_width
        self.num_classes=num_classes

    # Define a simple sequential model
    def build(self, hp):

        hp_dropout = hp.Float('dropout', min_value=0.01, max_value=0.30, step=0.1)
        hp_kernel = hp.Int('kernel', min_value=3, max_value=9, step=2)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model = Sequential([
            layers.Conv2D(32, hp_kernel, padding='same', activation='relu',
                          input_shape=(self.img_height, self.img_width, 3)),
            layers.MaxPooling2D(),
            layers.Dropout(hp_dropout),
            layers.Conv2D(64, hp_kernel, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(hp_dropout),
            layers.Conv2D(128, hp_kernel, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(hp_dropout),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.num_classes)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate, epsilon=0.1),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

# resize and normalize (0, 1] images
def resize_and_rescale(img, img_height=128, img_width=128): 
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, dtype=tf.float32) / 255.

    return img

# image augmentation function
def augment_image(img, img_height=128, img_width=128, rotate=0, Hflip=False, Vflip=False, brightness=0, zoom=0, contrast=0, gauss=0):
    # adapatation from https://stackoverflow.com/questions/65475057/keras-data-augmentation-pipeline-for-image-segmentation-dataset-image-and-mask

    # rotation in 360° steps
    if rotate != 0:
        rot_factor = tf.cast(tf.random.uniform(shape=[], minval=-rotate, maxval=rotate, dtype=tf.int32), tf.float32)
        angle = np.pi/360*rot_factor
        img = tfa.image.rotate(img, angle)

    # zoom in a bit
    if zoom != 0 and (tf.random.uniform(()) > 0.5):
        # use original image to preserve high resolution
        img = tf.image.central_crop(img, zoom)
        # resize
        img = tf.image.resize(img, [img_height, img_width])
    
    # apply gaussian filter
    if gauss != 0:
        sigma = random.uniform(0, gauss)
        img = tfa.image.gaussian_filter2d(img, sigma=sigma)

    # random brightness adjustment illumination
    if brightness != 0:
        img = tf.image.random_brightness(img, brightness)
    # random contrast adjustment
    if contrast != 0:
        img = tf.image.random_contrast(img, 1-contrast, 1+2*contrast)

    # reset all image values between 0 and 1
    img = tf.clip_by_value(img, 0.0, 1.0)

    # flipping random horizontal 
    if tf.random.uniform(()) > 0.5 and Hflip:
        img = tf.image.flip_left_right(img)
    # or vertical
    if tf.random.uniform(()) > 0.5 and Vflip:
        img = tf.image.flip_up_down(img)

    return img

def prepare(ds, augment=False, params={}):
  # Resize and rescale all datasets.
  img_height=params["img_height"]
  img_width=params["img_width"]
  ds = ds.map(lambda x, y: (resize_and_rescale(x, img_height=img_height, img_width=img_width), y), 
              num_parallel_calls=AUTOTUNE)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (augment_image(x, **params), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)

def show_samples(images, labels, class_names, batch_size):
    plt.figure(figsize=(10, 10))
    for i in range(batch_size):
      ax = plt.subplot(3, 3, i + 1)
      img_to_show = images[i].numpy()
      img_to_show = img_to_show * 255.
      img_to_show = img_to_show.astype("uint8")
      plt.imshow(img_to_show)
      plt.title(class_names[labels[i]])
      plt.axis("off")
    plt.show()

    return