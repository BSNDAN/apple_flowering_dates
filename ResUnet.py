import tensorflow as tf 
from tensorflow import keras


def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Activation("relu")(x)
    return x


def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = keras.layers.Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = keras.layers.Conv2D(num_filters, 3, padding="same", strides=1)(x)

    """ Shortcut Connection (Identity Mapping) """
    s = keras.layers.Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

    """ Addition """
    x = x + s
    return x


def build_resunet(input_shape):
    """ RESUNET Architecture """

    inputs = keras.layers.Input(input_shape)

    """ Endoder 1 """
    x = keras.layers.Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = keras.layers.Conv2D(64, 3, padding="same", strides=1)(x)
    s = keras.layers.Conv2D(64, 1, padding="same")(inputs)
    s1 = x + s

    """ Encoder 2, 3 """
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)

    """ Bridge """
    #b = residual_block(s3, 512, strides=2)
    
    """ Classifier """
    F = keras.layers.Flatten()(s3)
    D = keras.layers.Dense(10, activation='relu')(F)
    outputs = keras.layers.Dense(1)(D)
    opt = keras.optimizers.Adam(learning_rate=0.01)

    """ Model """
    model = keras.models.Model(inputs, outputs, name="RESUNET")

    return model