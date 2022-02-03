from keras import Sequential, layers
from tensorflow import losses

# Define a simple sequential model
def create_model(img_height, img_width, num_classes):
  data_augmentation = Sequential([
      layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
      layers.RandomRotation(0.2),
      layers.RandomZoom(0.2),
    ]
  )
  model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.15),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes)
  ])
  model.compile(
    optimizer='adam',
    loss = losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  return model