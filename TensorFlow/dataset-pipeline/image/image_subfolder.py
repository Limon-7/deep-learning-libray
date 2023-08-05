import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import pathlib


layers = keras.layers
# ImageDataGenerator = keras.preprocessing.image

#HYPERPARAMETERS
BATCH_SIZE = 32
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCH=5
img_height = 28
img_width = 28
batch_size = 2

model = keras.Sequential(
    [
        layers.Input((28, 28, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
)
data_dir = "TensorFlow/dataset-pipeline/image/data/mnist_subfolders/"
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
   data_dir,
    labels="inferred",
    label_mode="int", 
    #class_names=['0','1'...]
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle= True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int", 
    #class_names=['0','1'...]
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle= True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


ds_train = ds_train.map(augment)

# custom loop

model.compile(
    optimizer=keras.optimizers.Adam(0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(ds_train, epochs=EPOCH, verbose=2)


print("using DatatGenerator::::")

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=False,
    vertical_flip=False,
    data_format="channels_last",
    validation_split=0.0,
    dtype=tf.float32,
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=True,
    subset="training",
    seed=123,
)


def training():
    pass


# Custom Loops
for epoch in range(10):
    num_batches = 0

    for x, y in ds_train:
        num_batches += 1

        # do training
        training()

        if num_batches == 25:  # len(train_dataset)/batch_size
            break

# Redo model.compile to reset the optimizer states
model.compile(
    optimizer=keras.optimizers.Adam(.001),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

# using model.fit (note steps_per_epoch)
model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=25,
    verbose=2,
    # if we had a validation generator:
    # validation_data=validation_generator,
    # valiation_steps=len(validation_set)/batch_size),
)