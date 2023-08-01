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
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print(x_train.shape)

class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size = 3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()
    
    def call(self, input_tensor, training = False):
        x = self.conv(input_tensor)
        x = self.bn(x, training = training)
        x = tf.nn.relu(x)
        return x

sequential_model = keras.models.Sequential([
    CNNBlock(32),
    CNNBlock(64),
    CNNBlock(128),
    layers.Flatten(),
    layers.Dense(10)
])

history = sequential_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate= LEARNING_RATE),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    metrics = ["accuracy"]
)

sequential_model.fit(x_train, y_train, epochs = 1, batch_size= BATCH_SIZE, verbose=2)
sequential_model.evaluate(x_test, y_test, batch_size= BATCH_SIZE, verbose = 2)
print(sequential_model.summary())

class ResBlock(layers.Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.cnn1 = CNNBlock(channels[0])
        self.cnn2 = CNNBlock(channels[1])
        self.cnn3 = CNNBlock(channels[2])
        self.pooling = layers.MaxPooling2D()
        self.identity_mapping = layers.Conv2D(channels[1], kernel_size= 3, padding='same')
    
    def call(self, input_tensor, training = False):
        x = self.cnn1(input_tensor, training= training)
        x =self.cnn2(x, training= training)
        x = self.cnn3(x+self.identity_mapping(input_tensor), training = training)
        return self.pooling(x)
    
class ResNetLikeModel(keras.Model):
    def __init__(self, num_classes = 10):
        super(ResNetLikeModel, self).__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training = False):
        x =  self.block1(input_tensor, training=training)
        x =  self.block2(x, training=training)
        x =  self.block3(x, training=training)
        x = self.pool(x,  training=training)
        return self.classifier(x)
    
    def model(self):
        x = keras.Input(shape=(28, 28, 1))
        print(x)
        return keras.Model(inputs=[x], outputs=self.call(x))
    
model = ResNetLikeModel(num_classes= 10)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2)
# model.evaluate(x_test, y_test, batch_size= BATCH_SIZE, verbose = 2)
print(model.model().summary())