import tensorflow as tf
import SysproClassifyModel as SCM
import logging 
import os
import pathlib
import glob
import numpy as np
import json
import time

# This two lines are here just to silence Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

json_dir = pathlib.Path('./save/data.json')
with open(json_dir) as json_file:
    data = json.load(json_file)

save_dir = data["save_path"]

print('\nSTARTING \n')
print("Network trained on {}".format(data["training_date"]))

IMG_HEIGHT = data["img_height"]
IMG_WIDTH = data["img_width"]
class_names = data["classes"]
print(class_names)
NUM_CLASSES = len(class_names)

model = tf.keras.models.load_model(save_dir)

image_list = []
labels = np.zeros(200)
predictions = np.zeros(200)
labels[0:50] = 1
labels[50:100] = 3
labels[100:150] = 0
labels[150:200] = 2
i = 0

for filename in glob.glob('./imgs_meat/test/*.jpg'):

    img = tf.keras.utils.load_img(filename, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img) / 255.
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    start_time = time.time()
    predict = model.predict(img_array)
    execution_time = time.time() - start_time

    score = tf.nn.softmax(predict[0])
    predictions[i] = np.argmax(predict[0])

    print("The image N {} most likely belongs to {} with a {:.2f} percent confidence. It took {:.0f} miliseconds to predict"
          .format(i,
                  class_names[np.argmax(score)],
                  100 * np.max(score),
                  1000 * execution_time
                  ), end="\r", flush=True
          )

    i = i + 1

print(tf.math.confusion_matrix(
    labels, predictions, num_classes=None, weights=None, dtype=tf.dtypes.int32,
    name=None))