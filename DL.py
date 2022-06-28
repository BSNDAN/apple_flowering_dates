import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

###################################################  Gestion data ###################################################



def splitter(jeu:str, mois:str, data:pd.DataFrame):
    #  Pour un mois données, sépare les images (inputs) et les dates de floraison (output) #

    d_ = data[data.split == jeu].reset_index(drop=True)
    
    X_ = d_[mois] #récupérer le jeu correspondant dans le df pandas
    X_data = np.array([X_[idx] for idx in X_.index]) #transformer series(100,100,3) en np(len,100,100,3)
    
    y_data = d_["jourF"]
        
    print(f"{jeu} done")
    print("----------------")
    return X_data, y_data




################################################### ResUnet ###################################################


def batchnorm_relu(inputs):
    # Batch Normalization & ReLU #
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Activation("relu")(x)
    return x


def residual_block(inputs, num_filters, strides=1):
    # Convolutional Layers #
    x = batchnorm_relu(inputs)
    x = keras.layers.Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = keras.layers.Conv2D(num_filters, 3, padding="same", strides=1)(x)

    # Shortcut Connection (Identity Mapping) #
    s = keras.layers.Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

    # Addition #
    x = x + s
    return x


def build_resunet(input_shape):
    # RESUNET Architecture #

    inputs = keras.layers.Input(input_shape)

    # Endoder 1 #
    x = keras.layers.Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = keras.layers.Conv2D(64, 3, padding="same", strides=1)(x)
    s = keras.layers.Conv2D(64, 1, padding="same")(inputs)
    s1 = x + s

    # Encoder 2, 3 #
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)

    # Bridge #
    #b = residual_block(s3, 512, strides=2)
    
    # Classifier #
    F = keras.layers.Flatten()(s3)
    D = keras.layers.Dense(10, activation='relu')(F)
    outputs = keras.layers.Dense(1)(D)
    opt = keras.optimizers.Adam(learning_rate=0.01)

    # Model #
    model = keras.models.Model(inputs, outputs, name="RESUNET")

    return model



################################################### Unet ###################################################


def Unet(X_train):

    inputs = keras.layers.Input(X_train.shape[1:])

    c1 = keras.layers.BatchNormalization()(inputs)
    c1 = keras.layers.Conv2D(16, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(c1)
    c1 = keras.layers.BatchNormalization()(c1)
    c1 = keras.layers.Conv2D(16, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(c1)
    p1 = keras.layers.MaxPooling2D(2,2)(c1)

    c2 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(p1)
    c2 = keras.layers.BatchNormalization()(c2)
    c2 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(c2)
    p2 = keras.layers.MaxPooling2D(2,2)(c2)

    c3 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(p2)
    c3 = keras.layers.BatchNormalization()(c3)
    c3 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(c3)
    p3 = keras.layers.MaxPooling2D(2,2)(c3)

    c4 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(p3)
    c4 = keras.layers.BatchNormalization()(c4)
    c4 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(c4)
    p4 = keras.layers.MaxPooling2D(2,2)(c4)

    c5 = keras.layers.Conv2D(256, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(p4)
    c5 = keras.layers.BatchNormalization()(c5)
    c5 = keras.layers.Conv2D(256, (3,3), activation='selu', kernel_initializer="he_normal", padding = "same")(c5)

    F = keras.layers.Flatten()(c5)
    D = keras.layers.Dense(10)(F)
    D = keras.layers.Dropout(.1)(D)

    outputs = keras.layers.Dense(1)(D)
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model