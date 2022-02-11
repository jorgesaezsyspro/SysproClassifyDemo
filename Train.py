import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

import logging 
import os

import SysproClassifyModel as SCM
import pathlib

# This two lines are here just to silence Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Let's get started!
print('\nSTARTING \n')

# Setting some CONSTANTS 
AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 20
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CLASSES = 4
EPOCHS = 35

# Setting training image directories
data_dir = pathlib.Path('./imgs/train')
image_count = len(list(data_dir.glob('*/*.jpg')))
# And save directorie
save_dir = pathlib.Path('./save/cp_classify')

# Here we create the training image dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.5,
  subset="training",
  seed=42,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# And here the validation image dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.5,
  subset="validation",
  seed=42,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# Checking class names
class_names = train_ds.class_names
print(class_names)
NUM_CLASSES = len(class_names)

################################################################

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
  layers.Rescaling(1./255)
])

def data_augmentation(img, brightness=0.8, contrast=0.8):
  img = tf.image.random_brightness(img, brightness)
  img = tf.image.random_contrast(img, 1-contrast, 1+2*contrast)

  # flipping random horizontal 
  if tf.random.uniform(()) > 0.5:
    img = tf.image.flip_left_right(img)
  # or vertical
  if tf.random.uniform(()) > 0.5:
    img = tf.image.flip_up_down(img)

  return img


def prepare(ds, augment=False):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

  # Batch all datasets.
  # ds = ds.batch(BATCH_SIZE)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)


# Cache image files
AUTOTUNE = tf.data.AUTOTUNE
train_ds_prep = prepare(train_ds, augment=True)
val_ds = prepare(val_ds)
# test_ds = prepare(test_ds)

# Get some sample images from the training dataset to take a look
image_batch, labels_batch = next(iter(train_ds))
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img_to_show = images[i].numpy().astype("uint8")
    img_to_show = data_augmentation(img_to_show)
    plt.imshow(img_to_show)
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.draw()
plt.waitforbuttonpress(3)

# Create the model
model = SCM.classify_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)

# Take a look at the model structure
model.summary()

# Define some callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, save_weights_only=True, verbose=1)

# Fitting...
history = model.fit(
  train_ds_prep,
  validation_data=val_ds,
  epochs = EPOCHS,
  callbacks=[cp_callback]
)

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


# model = SysproClassifyModel.create_model()
# model.load_weights('./checkpoints/save')

# image_list = []
# labels = np.zeros(200)
# predictions = np.zeros(200)
# labels[0:50] = 1
# labels[50:100] = 3
# labels[100:150] = 0
# labels[150:200] = 2
# i=0

# for filename in glob.glob('./*.jpg'): #assuming png

#   img = tf.keras.utils.load_img(filename, target_size=(IMG_HEIGHT, IMG_WIDTH))
#   img_array = tf.keras.utils.img_to_array(img)
#   img_array = tf.expand_dims(img_array, 0) # Create a batch

#   predict = model.predict(img_array)
#   score = tf.nn.softmax(predict[0])
#   predictions[i] = np.argmax(predict[0])

#   #print("This image most likely belongs to {} with a {:.2f} percent confidence." .format(class_names[np.argmax(score)], 100 * np.max(score)))

#   i = i + 1

# print(tf.math.confusion_matrix(
#     labels, predictions, num_classes=None, weights=None, dtype=tf.dtypes.int32,
#     name=None))

# now = time.time()

# filename = './test/burger_meat_test_0.jpg' #assuming png
# img = tf.keras.utils.load_img(filename, target_size=(IMG_HEIGHT, IMG_WIDTH))
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
# predict = model.predict(img_array)
# predictions = np.argmax(predict)

# print(predictions)
# print(time.time() - now)

print('\n END \n')