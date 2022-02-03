import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import glob
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import SysproClassifyModel
import pathlib

print('\nSTARTING \n')

data_dir = pathlib.Path('./imgs/train')
image_count = len(list(data_dir.glob('*/*.jpg')))

BATCH_SIZE = 3
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 4
EPOCHS = 25

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="training",
  seed=42,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="validation",
  seed=42,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

model = SysproClassifyModel.create_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs = EPOCHS
)

model.save_weights('./checkpoints/save')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model = SysproClassifyModel.create_model()
model.load_weights('./checkpoints/save')

image_list = []
labels = np.zeros(200)
predictions = np.zeros(200)
labels[0:50] = 1
labels[50:100] = 3
labels[100:150] = 0
labels[150:200] = 2
i=0

for filename in glob.glob('./*.jpg'): #assuming png

  img = tf.keras.utils.load_img(filename, target_size=(IMG_HEIGHT, IMG_WIDTH))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predict = model.predict(img_array)
  score = tf.nn.softmax(predict[0])
  predictions[i] = np.argmax(predict[0])

  #print("This image most likely belongs to {} with a {:.2f} percent confidence." .format(class_names[np.argmax(score)], 100 * np.max(score)))

  i = i + 1

print(tf.math.confusion_matrix(
    labels, predictions, num_classes=None, weights=None, dtype=tf.dtypes.int32,
    name=None))

now = time.time()

filename = './test/burger_meat_test_0.jpg' #assuming png
img = tf.keras.utils.load_img(filename, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predict = model.predict(img_array)
predictions = np.argmax(predict)

print(predictions)
print(time.time() - now)

print('\n END \n')