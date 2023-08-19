from tensorflow.keras.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from tensorflow.keras import Model
import typing

def LeNet(input_shape: typing.Tuple[int], classes: int = 1000) -> Model:
    # convert input shape into tensor
    X_input = Input(input_shape)

    # layer 1
    X = Conv2D(
        filters = 6,
        kernel_size = (5, 5),
        strides = (1, 1),
        activation = "tanh",
        padding = "valid",
    )(X_input)
    X = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid")(X)

    # layer 2
    X = Conv2D(
        filters = 16,
        kernel_size = (5, 5),
        strides = (1, 1),
        activation = "tanh",
        padding = "valid",
    )(X)
    X = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid")(X)

    # layer 3
    X = Conv2D(
        filters = 120,
        kernel_size = (5, 5),
        strides = (1, 1),
        activation = "tanh",
        padding = "valid",
    )(X)

    # layer 4
    X = Flatten()(X)
    X = Dense(units = 84, activation = "tanh")(X)

    # layer 5 (classification layer)
    X = Dense(units = classes, activation = "softmax")(X)

    model = Model(inputs = X_input, outputs = X, name = "LeNet5")
    return model