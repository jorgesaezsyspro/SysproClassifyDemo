import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

import logging 
import os

import SysproClassifyModel as SCM
import pathlib

import json
import datetime

import keras_tuner as kt


# This two lines are here just to silence Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Let's get started!
print('\nSTARTING \n')

# Setting some CONSTANTS 
AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 5
IMG_HEIGHT = 256
IMG_WIDTH = IMG_HEIGHT
EPOCHS = 15

# Setting training image directories
data_dir = pathlib.Path('./imgs_meat/train')
image_count = len(list(data_dir.glob('*/*.jpg')))

# And save directories
save_dir = pathlib.Path('./save/cp_classify.h5')
json_dir = pathlib.Path('./save/data.json')

# Here we create the training image dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="training",
  seed=42,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# And here the validation image dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="validation",
  seed=42,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# Checking class names
class_names = train_ds.class_names
print(class_names)
NUM_CLASSES = len(class_names)

# Arguments for the image augmentation function
augment_params = {
    "img_height": IMG_HEIGHT,
    "img_width": IMG_WIDTH,
    "contrast": 0.15,
    "brightness": 0.15,
    "Hflip": True,
    "Vflip": True,
    "rotate": 30,
    "zoom": 0.9,
    "gauss": 15.0
}

# Prepare image files
train_ds = SCM.prepare(train_ds, augment=True, params=augment_params)
val_ds = SCM.prepare(val_ds, params=augment_params)

# Get some sample images from the training dataset to take a look
images, labels = next(iter(train_ds))
SCM.show_samples(images, labels, class_names, BATCH_SIZE)

# Create the model
hp=kt.HyperParameters()
my_hyper_model = SCM.MyHyperModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_classes=NUM_CLASSES)
tuner = kt.BayesianOptimization(my_hyper_model,
                                objective='val_accuracy',
                                max_trials=10,
                                directory='trials',
                                project_name='classify',
                                overwrite=True)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
tuner.search(train_ds,
             validation_data=val_ds,
             epochs=EPOCHS,
             callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Take a look at the model structure
# model.summary()

# Define some callbacks
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, save_weights_only=True, verbose=1)

# Fitting...
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=[]
                    )

# Save the model
model.save(filepath=save_dir, overwrite=True)

# Show training data
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

# Save some data for the Test part
data = {
  "training_date" : str(datetime.datetime.now()),
  "save_path"     : save_dir,
  "classes"       : class_names,
  "img_height"    : IMG_HEIGHT,
  "img_width"     : IMG_WIDTH,
  "augment_params": augment_params
}
with open(json_dir, 'w') as f:
    json.dump(data, f)

print('\n END \n')