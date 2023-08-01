import tensorflow as tf
from tensorflow import keras
import pandas as pd

layers = keras.layers
regularizers = keras.regularizers
mnist = keras.datasets.mnist

#HYPERPARAMETERS
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

print(x_train.shape)

class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64)
        self.dense2 = layers.Dense(num_classes)
    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)
    
model = MyModel(num_classes= 10)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=2)

# model.evaluate(x_test, y_test, batch_size= BATCH_SIZE, verbose = 2)
