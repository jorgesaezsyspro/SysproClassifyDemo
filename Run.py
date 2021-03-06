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

save_dir = pathlib.Path('./save/cp_classify.h5')

print('\nSTARTING \n')
print("Network trained on {}".format(data["training_date"]))

IMG_HEIGHT = data["img_height"]
IMG_WIDTH = data["img_width"]
class_names = data["classes"]
print(class_names)
NUM_CLASSES = len(class_names)

# Load model
model = tf.keras.models.load_model(save_dir)
# TODO Check that loaded model is ok

# TODO Connect to camera
# TODO Check

# TODO Connect to PLC
# TODO Check

# Loop until close:
while(True):
    # TODO Wait for trigger input
    # TODO Trigger camera

    # TODO Load img to model
    image = []
    # Predict
    img = tf.keras.utils.load_img(image, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img) / 255.
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    # Get results
    start_time = time.time()
    predict = model.predict(img_array)
    execution_time = time.time() - start_time

    score = np.max(tf.nn.softmax(predict[0])) * 100
    prediction = class_names[np.argmax(score)]

    # TODO Make a decision

    break