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
from tensorflow.python.framework import ops


# layer = tf.keras.layers.ReLU
# tf.keras.layers.BatchNormalization

# sobel horizon weight constraint and fix
class HBetweenAndFix(Constraint):
    def __init__(self, min_value1=0.0, max_value1=1.0, min_value2=-1.0, max_value2=0.0):
        self.min_value1 = min_value1
        self.max_value1 = max_value1
        self.min_value2 = min_value2
        self.max_value2 = max_value2

    def __call__(self, w):
        # first get weight shape
        w_shape = w.shape
        if w_shape.rank is None or w_shape.rank != 4:
            raise ValueError(
                'The weight tensor must have rank 4. '
                f'Received weight tensor with shape: {w_shape}')

        height, width, channels, kernels = w_shape
        if height != 3 or width != 3:
            raise ValueError(
                'mask shape must be 3x3. '
                f'Received weight tensor with shape: {height} x {width}')
        hw0 = K.clip(w[0, 0:3, 0:channels, 0:kernels], self.min_value1, self.max_value1)
        hw2 = K.clip(w[2, 0:3, 0:channels, 0:kernels], self.min_value2, self.max_value2)
        K.set_value(w[1, 0, 0:channels, 0:kernels],np.zeros((channels, kernels)))
        K.set_value(w[1, 1, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        K.set_value(w[1, 2, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        hw1 = tf.stack([w[1, 0, 0:channels, 0:kernels], w[1, 1, 0:channels, 0:kernels], w[1, 2, 0:channels, 0:kernels]])
        w_stack = tf.stack([hw0, hw1, hw2])

        """
        for i in range(0, channels):
            for j in range(0, kernels):
                K.clip(w[0, 0:2, i, j], self.min_value1, self.max_value1)
                K.clip(w[2, 0:2, i, j], self.min_value2, self.max_value2)
                K.set_value(w[1, 0, i, j], 0)
                K.set_value(w[1, 1, i, j], 0)
                K.set_value(w[1, 2, i, j], 0)
        """
        #tf.print(w_stack[0:3,0:3,0,0])
        return w_stack

    def get_config(self):
        return {'min_value1': self.min_value1,
                'max_value1': self.max_value1,
                'min_value2': self.min_value2,
                'max_value2': self.max_value2,
                }


# sobel vertical weight constraint and fix
class VBetweenAndFix(Constraint):
    def __init__(self, min_value1=0., max_value1=1., min_value2=-1., max_value2=0.):
        self.min_value1 = min_value1
        self.max_value1 = max_value1
        self.min_value2 = min_value2
        self.max_value2 = max_value2

    def __call__(self, w):
        # first get weight shape
        w_shape = w.shape
        if w_shape.rank is None or w_shape.rank != 4:
            raise ValueError(
                'The weight tensor must have rank 4. '
                f'Received weight tensor with shape: {w_shape}')

        height, width, channels, kernels = w_shape
        if height != 3 or width != 3:
            raise ValueError(
                'mask shape must be 3x3. '
                f'Received weight tensor with shape: {height} x {width}')
        vw0 = K.clip(w[0:3, 0, 0:channels, 0:kernels], self.min_value1, self.max_value1)
        vw2 = K.clip(w[0:3, 2, 0:channels, 0:kernels], self.min_value2, self.max_value2)
        K.set_value(w[0, 1, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        K.set_value(w[1, 1, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        K.set_value(w[2, 1, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        vw1 = tf.stack([w[0, 1, 0:channels, 0:kernels], w[1, 1, 0:channels, 0:kernels], w[2, 1, 0:channels, 0:kernels]])
        w_stack = tf.stack([vw0, vw1, vw2], axis=1)
        """
        for i in range(0, channels):
            for j in range(0, kernels):
                K.clip(w[0:2, 0, i, j], self.min_value1, self.max_value1)
                K.clip(w[0:2, 2, i, j], self.min_value2, self.max_value2)
                K.set_value(w[0, 1, i, j], 0)
                K.set_value(w[1, 1, i, j], 0)
                K.set_value(w[2, 1, i, j], 0)
        """
        return w_stack

    def get_config(self):
        return {'min_value1': self.min_value1,
                'max_value1': self.max_value1,
                'min_value2': self.min_value2,
                'max_value2': self.max_value2,
                }


# sobel positive diagnal weight constraint and fix
class PDBetweenAndFix(Constraint):
    def __init__(self, min_value1=0., max_value1=1., min_value2=-1., max_value2=0.):
        self.min_value1 = min_value1
        self.max_value1 = max_value1
        self.min_value2 = min_value2
        self.max_value2 = max_value2

    def __call__(self, w):
        # first get weight shape
        w_shape = w.shape
        if w_shape.rank is None or w_shape.rank != 4:
            raise ValueError(
                'The weight tensor must have rank 4. '
                f'Received weight tensor with shape: {w_shape}')

        height, width, channels, kernels = w_shape
        if height != 3 or width != 3:
            raise ValueError(
                'mask shape must be 3x3. '
                f'Received weight tensor with shape: {height} x {width}')
        pw00 = K.clip(w[0, 0, 0:channels, 0:kernels], self.min_value1, self.max_value1)
        pw01 = K.clip(w[0, 1, 0:channels, 0:kernels], self.min_value1, self.max_value1)
        pw10 = K.clip(w[1, 0, 0:channels, 0:kernels], self.min_value1, self.max_value1)
        pw12 = K.clip(w[1, 2, 0:channels, 0:kernels], self.min_value2, self.max_value2)
        pw21 = K.clip(w[2, 1, 0:channels, 0:kernels], self.min_value2, self.max_value2)
        pw22 = K.clip(w[2, 2, 0:channels, 0:kernels], self.min_value2, self.max_value2)
        K.set_value(w[0, 2, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        K.set_value(w[1, 1, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        K.set_value(w[2, 0, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        pw0 = tf.stack([pw00, pw01, w[0, 2, 0:channels, 0:kernels]])
        pw1 = tf.stack([pw10, w[1, 1, 0:channels, 0:kernels], pw12])
        pw2 = tf.stack([w[2, 0, 0:channels, 0:kernels], pw21, pw22])
        w_stack = tf.stack([pw0, pw1, pw2])

        """
        for i in range(0, channels):
            for j in range(0, kernels):
                K.clip(w[0, 0, i, j], self.min_value1, self.max_value1)
                K.clip(w[0, 1, i, j], self.min_value1, self.max_value1)
                K.clip(w[1, 0, i, j], self.min_value1, self.max_value1)
                K.clip(w[1, 2, i, j], self.min_value2, self.max_value2)
                K.clip(w[2, 1, i, j], self.min_value2, self.max_value2)
                K.clip(w[2, 2, i, j], self.min_value2, self.max_value2)
                K.set_value(w[0, 2, i, j], 0)
                K.set_value(w[1, 1, i, j], 0)
                K.set_value(w[2, 0, i, j], 0)
        """
        return w_stack

    def get_config(self):
        return {'min_value1': self.min_value1,
                'max_value1': self.max_value1,
                'min_value2': self.min_value2,
                'max_value2': self.max_value2,
                }


# sobel nagative diagnal weight constraint and fix
class NDBetweenAndFix(Constraint):
    def __init__(self, min_value1=0., max_value1=1., min_value2=-1., max_value2=0.):
        self.min_value1 = min_value1
        self.max_value1 = max_value1
        self.min_value2 = min_value2
        self.max_value2 = max_value2

    def __call__(self, w):
        # first get weight shape
        w_shape = w.shape
        if w_shape.rank is None or w_shape.rank != 4:
            raise ValueError(
                'The weight tensor must have rank 4. '
                f'Received weight tensor with shape: {w_shape}')

        height, width, channels, kernels = w_shape
        if height != 3 or width != 3:
            raise ValueError(
                'mask shape must be 3x3. '
                f'Received weight tensor with shape: {height} x {width}')
        nw01 = K.clip(w[0, 1, 0:channels, 0:kernels], self.min_value1, self.max_value1)
        nw02 = K.clip(w[0, 2, 0:channels, 0:kernels], self.min_value1, self.max_value1)
        nw12 = K.clip(w[1, 2, 0:channels, 0:kernels], self.min_value1, self.max_value1)
        nw10 = K.clip(w[1, 0, 0:channels, 0:kernels], self.min_value2, self.max_value2)
        nw20 = K.clip(w[2, 0, 0:channels, 0:kernels], self.min_value2, self.max_value2)
        nw21 = K.clip(w[2, 1, 0:channels, 0:kernels], self.min_value2, self.max_value2)
        K.set_value(w[0, 0, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        K.set_value(w[1, 1, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        K.set_value(w[2, 2, 0:channels, 0:kernels], np.zeros((channels, kernels)))
        nw0 = tf.stack([w[0, 0, 0:channels, 0:kernels], nw01, nw02])
        nw1 = tf.stack([nw10, w[1, 1, 0:channels, 0:kernels], nw12])
        nw2 = tf.stack([nw20, nw21, w[2, 2, 0:channels, 0:kernels]])
        w_stack = tf.stack([nw0, nw1, nw2])
        """
        for i in range(0, channels):
            for j in range(0, kernels):
                K.clip(w[0, 1, i, j], self.min_value1, self.max_value1)
                K.clip(w[0, 2, i, j], self.min_value1, self.max_value1)
                K.clip(w[1, 2, i, j], self.min_value1, self.max_value1)
                K.clip(w[1, 0, i, j], self.min_value2, self.max_value2)
                K.clip(w[2, 0, i, j], self.min_value2, self.max_value2)
                K.clip(w[2, 1, i, j], self.min_value2, self.max_value2)
                K.set_value(w[0, 0, i, j], 0)
                K.set_value(w[1, 1, i, j], 0)
                K.set_value(w[2, 2, i, j], 0)
        """
        return w_stack

    def get_config(self):
        return {'min_value1': self.min_value1,
                'max_value1': self.max_value1,
                'min_value2': self.min_value2,
                'max_value2': self.max_value2,
                }


@tf.custom_gradient
def threshold2binary(x, t=0.8):
    def _threshold2binary_grad(grad):

        #return grad_cus * grad
        return 0 * grad
        #return 1 / (1 + tf.exp(-(x-thresholdsobel_extend))) * (1 - (1 / (1 + tf.exp(-(x-thresholdsobel_extend))))) * grad #1 * grad
    x = tf.abs(x)
    thresholdsobel = tf.reduce_max(x, [1,2])
    thresholdsobel = tf.expand_dims(thresholdsobel, 1)
    thresholdsobel = tf.expand_dims(thresholdsobel, 1)
    x_shape = x.shape
    batch, height, width, channels = x_shape
    thresholdsobel_extend = tf.tile(thresholdsobel, [1, height, width, 1])
    #tf.print(thresholdsobel_extend)
    """
    x_shape = x.shape
    if x_shape.rank is None or x_shape.rank != 4:
        raise ValueError(
            'The input tensor must have rank 4. '
            f'Received input tensor with shape: {x_shape}')

    batch, height, width, channels = x_shape
    """
    cond = tf.less(x, t*thresholdsobel_extend)
    #tf.print(cond[5, 0:4, 0:4, 50])
    x = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    #tf.print(x[5,0:4,0:4,50])
    #grad_cus = 1 / (1 + tf.exp(-(x-thresholdsobel_extend))) * (1 - (1 / (1 + tf.exp(-(x-thresholdsobel_extend)))))

    cond1 = tf.less(x, t * thresholdsobel_extend)
    # tf.print(cond[5, 0:4, 0:4, 50])
    grad_cus = tf.where(cond1, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return x, _threshold2binary_grad


get_custom_objects().update({'threshold2binary': layers.Activation(threshold2binary)})

"""
@tf.RegisterGradient("threshold2binaryGrad")
def _threshold2binary_grad(op, grad):
    return 1 * grad


#tf.compat.v1.disable_eager_execution()
a = tf.Variable(tf.constant([5., 4., 3., 2., 1.], dtype=tf.float32))
h = threshold2binary(a)

g = tf.compat.v1.get_default_graph()
with g.gradient_override_map({'Identity': 'threshold2binaryGrad'}):
    h = tf.identity(h, name="threshold2binary")
    #grad1 = tf.gradients(h, a)

"""
class Vgg16binary(keras.Model):
    def __init__(self, name="Vgg16binary", **kwargs):
        super(Vgg16binary, self).__init__()
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
        self.conv2D512relu4 = layers.Conv2D(512, 1, strides=1, padding="same", activation="relu")
        self.conv2D512relu5 = layers.Conv2D(512, 1, strides=1, padding="same", activation="relu")
        self.conv2D512relu6 = layers.Conv2D(512, 1, strides=1, padding="same", activation="relu")
        # self.activation2 = layers.Activation(activations.exponential)
        self.flatten1 = layers.Flatten()

        # another branch which incorporates binary features
        # shallow binary feature
        # Entry block
        # 分成4个卷积层，sobel
        #self.rescale2 = layers.Rescaling(1.0 / 255)

        self.conv2D32HB1 = layers.Conv2D(32, 3, strides=1, padding="same",
                                         kernel_constraint=HBetweenAndFix(),
                                         activation='relu')
        # x11 = layers.Activation(activations.tanh(x11))
        self.conv2D32VB1 = layers.Conv2D(32, 3, strides=1, padding="same",
                                         kernel_constraint=VBetweenAndFix(),
                                         activation='relu')
        # x12 = layers.Activation(activations.tanh(x12))
        self.conv2D32PB1 = layers.Conv2D(32, 3, strides=1, padding="same",
                                         kernel_constraint=PDBetweenAndFix(),
                                         activation='relu')
        # x13 = layers.Activation(activations.tanh(x13))
        self.conv2D32NB1 = layers.Conv2D(32, 3, strides=1, padding="same",
                                         kernel_constraint=NDBetweenAndFix(),
                                         activation='relu')
        # x14 = layers.Activation(activations.tanh(x14))

        # or to try average layer
        self.add1 = layers.Add()
        self.Batchnorm5 = layers.BatchNormalization()

        self.maxPolling5 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')
        self.conv2D64HB1 = layers.Conv2D(64, 3, strides=1, padding="same",
                                         kernel_constraint=HBetweenAndFix(),
                                         activation='relu')
        # x21 = layers.Activation(activations.tanh(x21))
        self.conv2D64VB1 = layers.Conv2D(64, 3, strides=1, padding="same",
                                         kernel_constraint=VBetweenAndFix(),
                                         activation='relu')
        # x22 = layers.Activation(activations.tanh(x22))
        self.conv2D64PB1 = layers.Conv2D(64, 3, strides=1, padding="same",
                                         kernel_constraint=PDBetweenAndFix(),
                                         activation='relu')
        # x23 = layers.Activation(activations.tanh(x23))
        self.conv2D64NB1 = layers.Conv2D(64, 3, strides=1, padding="same",
                                         kernel_constraint=NDBetweenAndFix(),
                                         activation='relu')
        # x24 = layers.Activation(activations.tanh(x24))

        self.add2 = layers.Add()
        self.Batchnorm6 = layers.BatchNormalization()

        self.maxPolling6 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')
        self.conv2D128HB1 = layers.Conv2D(128, 3, strides=1, padding="same",
                                          kernel_constraint=HBetweenAndFix(),
                                          activation='relu')
        # x31 = layers.Activation(activations.tanh(x31))
        self.conv2D128VB1 = layers.Conv2D(128, 3, strides=1, padding="same",
                                          kernel_constraint=VBetweenAndFix(),
                                          activation='relu')
        # x32 = layers.Activation(activations.tanh(x32))
        self.conv2D128PB1 = layers.Conv2D(128, 3, strides=1, padding="same",
                                          kernel_constraint=PDBetweenAndFix(),
                                          activation='relu')
        # x33 = layers.Activation(activations.tanh(x33))

        self.conv2D128NB1 = layers.Conv2D(128, 3, strides=1, padding="same",
                                          kernel_constraint=NDBetweenAndFix(),
                                          activation='relu')
        # x34 = layers.Activation(activations.tanh(x34))

        self.add3 = layers.Add()

        self.maxPolling7 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')

        # 4x4x128
        # thresholded customerized layer
        # x4 = activations.threshold2binary(x4)
        self.activation1 = layers.Activation(threshold2binary)
        #self.activation1 = layers.ThresholdedReLU(0.6)
        #self.activation1 = layers.Activation('sigmoid')
        self.flatten2 = layers.Flatten()
        # weighted
        #   x1 = 0.5*x1

        # concat
        self.concat1 = layers.Concatenate()

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
        x
        x = self.conv2D64relu2(x)
        #tf.print(x[5, 0:16, 0:16, 50])
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
        #x = self.conv2D512relu4(x)
        #x = self.conv2D512relu5(x)
        #x = self.conv2D512relu6(x)
        # x = self.activation2(x)
        x = self.flatten1(x)
        # weighted
        #    x = 0.5*x
        # tf.print(x[5,50:65])
        # another branch which incorporates binary features
        # shallow binary feature
        # Entry block
        # 分成4个卷积层，sobel
        #x1 = self.rescale2(inputs)
        x1 = inputs

        x11 = self.conv2D32HB1(x1)
        # x11 = layers.Activation(activations.tanh(x11))


        # or to try average layer
        x2 = x11
        #tf.print(x2[5, 0:32, 0:32, 20])
        #x2 = self.Batchnorm5(x2)

        x2 = self.maxPolling5(x2)
        x21 = self.conv2D64HB1(x2)
        #tf.print(x21[5, 0:16, 0:16, 50])
        # x21 = layers.Activation(activations.tanh(x21))


        x3 = x21

        #tf.print(x3[5, 0:16,0:16,50])
        #x3 = self.Batchnorm6(x3)

        x3 = self.maxPolling6(x3)


        #x4 = self.maxPolling7(x4)

        # 4x4x128
        # thresholded customerized layer
        # x4 = activations.threshold2binary(x4)

        #tf.print(x4[5, 0:4, 0:4, 50])
        x4 = self.activation1(x3)

        x4 = self.flatten2(x4)
        # weighted
        #   x1 = 0.5*x1
        #tf.print(x4[5, 50:65])
        # concat
        features = self.concat1([x, x4])

        # fully connected layers
        features = self.dense1(features)
        features = self.dropout1(features)
        features = self.dense2(features)
        features = self.dropout2(features)
        features = self.dense3(features)
        outputs = self.classifier(features)
        return outputs
