import keras.layers
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
from tensorflow import keras
from keras import layers
from keras import activations
from keras import utils
import matplotlib as mpl
import math
from keras.constraints import Constraint
import keras.backend as K
#from keras.utils.generic_utils import get_custom_objects
from keras.utils import get_custom_objects
import random
from keras.constraints import max_norm

class Vgg16(keras.Model):
    def __init__(self, name="Vgg16", **kwargs):
        super(Vgg16, self).__init__()
        #self.rescale1 = layers.Rescaling(1.0 / 255)
        self.conv2D64relu1 = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")
        self.conv2D64relu2 = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")
        self.conv2D64relu3 = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")
        self.Batchnorm1 = layers.BatchNormalization()

        self.maxPolling1 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')
        self.conv2D128relu1 = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")
        self.conv2D128relu2 = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")
        self.conv2D128relu3 = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")
        self.Batchnorm2 = layers.BatchNormalization()

        self.maxPolling2 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')
        self.conv2D256relu1 = layers.Conv2D(256, 3, strides=1, padding="same", activation="relu")
        self.conv2D256relu2 = layers.Conv2D(256, 3, strides=1, padding="same", activation="relu")
        self.conv2D256relu3 = layers.Conv2D(256, 3, strides=1, padding="same", activation="relu")
        self.Batchnorm3 = layers.BatchNormalization()

        self.maxPolling3 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')
        self.conv2D512relu1 = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")
        self.conv2D512relu2 = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")
        self.conv2D512relu3 = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")
        self.Batchnorm4 = layers.BatchNormalization()  # 4x4x512

        # x = layers.nv2D(512, 3, MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        #         # x = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")(x)
        #         # x = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")(x)
        #         # x = layers.Costrides=1, padding="same", activation="relu")(x)
        # x = layers.BatchNormalization()(x)

        self.maxPolling4 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')  # 2x2x512
        # self.activation2 = layers.Activation(activations.exponential)
        self.flatten1 = layers.Flatten()

               # fully connected layers
        self.dense1 = layers.Dense(4096, activation="sigmoid")
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(4096, activation="sigmoid")
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(10)
        self.classifier = layers.Softmax()

    def call(self, inputs):
        # inputs = keras.Input(shape=input_shape)

        # Entry block
        #x = self.rescale1(inputs)
        x = inputs
        x = self.conv2D64relu1(x)
        x = self.conv2D64relu2(x)
        x = self.conv2D64relu3(x)
        x = self.Batchnorm1(x)

        x = self.maxPolling1(x)
        x = self.conv2D128relu1(x)
        x = self.conv2D128relu2(x)
        x = self.conv2D128relu3(x)
        x = self.Batchnorm2(x)

        x = self.maxPolling2(x)
        x = self.conv2D256relu1(x)
        x = self.conv2D256relu2(x)
        x = self.conv2D256relu3(x)
        x = self.Batchnorm3(x)

        x = self.maxPolling3(x)
        x = self.conv2D512relu1(x)
        x = self.conv2D512relu2(x)
        x = self.conv2D512relu3(x)
        x = self.Batchnorm4(x)  # 4x4x512

        # x = layers.nv2D(512, 3, MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        #         # x = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")(x)
        #         # x = layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")(x)
        #         # x = layers.Costrides=1, padding="same", activation="relu")(x)
        # x = layers.BatchNormalization()(x)

        x = self.maxPolling4(x)  # 2x2x512
        # x = self.activation2(x)
        x = self.flatten1(x)
        features = x

        # fully connected layers
        features = self.dense1(features)
        features = self.dropout1(features)
        features = self.dense2(features)
        features = self.dropout2(features)
        features = self.dense3(features)
        outputs = self.classifier(features)
        return outputs
