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
def threshold2binary(x, t=0.6):
    def _threshold2binary_grad(grad):

        return 0 * grad
        #return grad_cus * grad
    x1 = tf.abs(x)
    thresholdsobel = tf.reduce_max(x1, [1,2])
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
    cond = tf.less(x1, t*thresholdsobel_extend)
    #tf.print(cond[5, 0:4, 0:4, 50])
    x1 = tf.where(cond, tf.zeros(tf.shape(x1)), tf.ones(tf.shape(x1)))

    #tf.print(x[5,0:4,0:4,50])
    #grad_cus = tf.exp(-(tf.abs(x)-thresholdsobel_extend)) / ((1 + tf.exp(-(tf.abs(x)-thresholdsobel_extend))) * (1 + tf.exp(-(tf.abs(x)-thresholdsobel_extend))))
    grad_cus = 1 / (1 + tf.exp(-1*(tf.abs(x) - thresholdsobel_extend))) * (1 - (1 / (1 + tf.exp(-1*(tf.abs(x) - thresholdsobel_extend)))))
    # cond1 = tf.less(x, t * thresholdsobel_extend)
    # grad_cus = tf.where(cond1, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return x1, _threshold2binary_grad


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
class ResNet34binary(keras.Model):
    def __init__(self, name="ResNet34binary", **kwargs):
        super(ResNet34binary, self).__init__()
        #self.rescale1 = layers.Rescaling(1.0 / 255)
        self.conv2D1 = layers.Conv2D(64, 7, strides=1, padding="same", activation="relu")

        # block 1  3x3 64 2  x3
        self.conv2D2 = layers.Conv2D(64, 3, strides=1, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(activations.relu)
        self.conv2D3 = layers.Conv2D(64, 3, strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.add1 = layers.Add()
        self.act2 = layers.Activation(activations.relu)
        self.conv2D4 = layers.Conv2D(64, 3, strides=1, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.Activation(activations.relu)
        self.conv2D5 = layers.Conv2D(64, 3, strides=1, padding="same")
        self.bn4 = layers.BatchNormalization()
        self.add2 = layers.Add()
        self.act4 = layers.Activation(activations.relu)
        self.conv2D6 = layers.Conv2D(64, 3, strides=1, padding="same")
        self.bn5 = layers.BatchNormalization()
        self.act5 = layers.Activation(activations.relu)
        self.conv2D7 = layers.Conv2D(64, 3, strides=1, padding="same")
        self.bn6 = layers.BatchNormalization()
        self.add3 = layers.Add()
        self.act6 = layers.Activation(activations.relu)

        # block 2 3x3 128 2   x4
        self.conv2D8 = layers.Conv2D(128, 3, strides=2, padding="same")
        self.bn7 = layers.BatchNormalization()
        self.act7 = layers.Activation(activations.relu)
        self.conv2D9 = layers.Conv2D(128, 3, strides=1, padding="same")
        self.bn8 = layers.BatchNormalization()
        self.sccov2D1 = layers.Conv2D(128, 1, strides=2, padding="same")
        self.add4 = layers.Add()
        self.act8 = layers.Activation(activations.relu)
        self.conv2D10 = layers.Conv2D(128, 3, strides=1, padding="same")
        self.bn9 = layers.BatchNormalization()
        self.act9 = layers.Activation(activations.relu)
        self.conv2D11 = layers.Conv2D(128, 3, strides=1, padding="same")
        self.bn10 = layers.BatchNormalization()
        self.add5 = layers.Add()
        self.act10 = layers.Activation(activations.relu)
        self.conv2D12 = layers.Conv2D(128, 3, strides=1, padding="same")
        self.bn11 = layers.BatchNormalization()
        self.act11 = layers.Activation(activations.relu)
        self.conv2D13 = layers.Conv2D(128, 3, strides=1, padding="same")
        self.bn12 = layers.BatchNormalization()
        self.add6 = layers.Add()
        self.act12 = layers.Activation(activations.relu)
        self.conv2D14 = layers.Conv2D(128, 3, strides=1, padding="same")
        self.bn13 = layers.BatchNormalization()
        self.act13 = layers.Activation(activations.relu)
        self.conv2D15 = layers.Conv2D(128, 3, strides=1, padding="same")
        self.bn14 = layers.BatchNormalization()
        self.add7 = layers.Add()
        self.act14 = layers.Activation(activations.relu)

        # block 3x3 256 2  x6
        self.conv2D16 = layers.Conv2D(256, 3, strides=2, padding="same")
        self.bn15 = layers.BatchNormalization()
        self.act15 = layers.Activation(activations.relu)
        self.conv2D17 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn16 = layers.BatchNormalization()
        self.sccov2D2 = layers.Conv2D(256, 1, strides=2, padding="same")
        self.add8 = layers.Add()
        self.act16 = layers.Activation(activations.relu)
        self.conv2D18 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn17 = layers.BatchNormalization()
        self.act17 = layers.Activation(activations.relu)
        self.conv2D19 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn18 = layers.BatchNormalization()
        self.add9 = layers.Add()
        self.act18 = layers.Activation(activations.relu)
        self.conv2D20 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn19 = layers.BatchNormalization()
        self.act19 = layers.Activation(activations.relu)
        self.conv2D21 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn20 = layers.BatchNormalization()
        self.add10 = layers.Add()
        self.act20 = layers.Activation(activations.relu)
        self.conv2D22 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn21 = layers.BatchNormalization()
        self.act21 = layers.Activation(activations.relu)
        self.conv2D23 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn22 = layers.BatchNormalization()
        self.add11 = layers.Add()
        self.act22 = layers.Activation(activations.relu)
        self.conv2D24 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn23 = layers.BatchNormalization()
        self.act23 = layers.Activation(activations.relu)
        self.conv2D25 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn24 = layers.BatchNormalization()
        self.add12 = layers.Add()
        self.act24 = layers.Activation(activations.relu)
        self.conv2D26 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn25 = layers.BatchNormalization()
        self.act25 = layers.Activation(activations.relu)
        self.conv2D27 = layers.Conv2D(256, 3, strides=1, padding="same")
        self.bn26 = layers.BatchNormalization()
        self.add13 = layers.Add()
        self.act26 = layers.Activation(activations.relu)

        # block 3x3 512 2  x3
        self.conv2D28 = layers.Conv2D(512, 3, strides=2, padding="same")
        self.bn27 = layers.BatchNormalization()
        self.act27 = layers.Activation(activations.relu)
        self.conv2D29 = layers.Conv2D(512, 3, strides=1, padding="same")
        self.bn28 = layers.BatchNormalization()
        self.sccov2D3 = layers.Conv2D(512, 1, strides=2, padding="same")
        self.add14 = layers.Add()
        self.act28 = layers.Activation(activations.relu)
        self.conv2D30 = layers.Conv2D(512, 3, strides=1, padding="same")
        self.bn29 = layers.BatchNormalization()
        self.act29 = layers.Activation(activations.relu)
        self.conv2D31 = layers.Conv2D(512, 3, strides=1, padding="same")
        self.bn30 = layers.BatchNormalization()
        self.add15 = layers.Add()
        self.act30 = layers.Activation(activations.relu)
        self.conv2D32 = layers.Conv2D(512, 3, strides=1, padding="same")
        self.bn31 = layers.BatchNormalization()
        self.act31 = layers.Activation(activations.relu)
        self.conv2D33 = layers.Conv2D(512, 3, strides=1, padding="same")
        self.bn32 = layers.BatchNormalization()
        self.add16 = layers.Add()
        self.act32 = layers.Activation(activations.relu)

        self.avep1 = layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='same')
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
        # self.activation1 = layers.ReLU(0.6, 0, 0.2)
        self.flatten2 = layers.Flatten()
        # weighted
        #   x1 = 0.5*x1

        # concat
        self.concat1 = layers.Concatenate()

        # fully connected layers
        self.dense = layers.Dense(10)
        self.classifier = layers.Softmax()

    def call(self, inputs):
        # inputs = keras.Input(shape=input_shape)

        # Entry block
        #x = self.rescale1(inputs)
        x = inputs
        x = self.conv2D1(x)
        x_temp = x

        # block 1
        x = self.conv2D2(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2D3(x)
        x = self.bn2(x)
        x = self.add1([x_temp, x])
        x = self.act2(x)
        x_temp = x
        x = self.conv2D4(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv2D5(x)
        x = self.bn4(x)
        x = self.add2([x_temp, x])
        x = self.act4(x)
        x_temp = x
        x = self.conv2D6(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.conv2D7(x)
        x = self.bn6(x)
        x = self.add3([x_temp, x])
        x = self.act6(x)
        temp_x = x

        # block 2
        x = self.conv2D8(x)
        x = self.bn7(x)
        x = self.act7(x)
        x = self.conv2D9(x)
        x = self.bn8(x)
        temp_x = self.sccov2D1(temp_x)
        x = self.add4([temp_x, x])
        x = self.act8(x)
        temp_x = x
        x = self.conv2D10(x)
        x = self.bn9(x)
        x = self.act9(x)
        x = self.conv2D11(x)
        x = self.bn10(x)
        x = self.add5([temp_x, x])
        x = self.act10(x)
        temp_x = x
        x = self.conv2D12(x)
        x = self.bn11(x)
        x = self.act11(x)
        x = self.conv2D13(x)
        x = self.bn12(x)
        x = self.add6([temp_x, x])
        x = self.act12(x)
        temp_x = x
        x = self.conv2D14(x)
        x = self.bn13(x)
        x = self.act13(x)
        x = self.conv2D15(x)
        x = self.bn14(x)
        x = self.add7([temp_x, x])
        x = self.act14(x)
        temp_x = x

        # block 3
        x = self.conv2D16(x)
        x = self.bn15(x)
        x = self.act15(x)
        x = self.conv2D17(x)
        x = self.bn16(x)
        temp_x = self.sccov2D2(temp_x)
        x = self.add8([temp_x, x])
        x = self.act16(x)
        temp_x = x
        x = self.conv2D18(x)
        x = self.bn17(x)
        x = self.act17(x)
        x = self.conv2D19(x)
        x = self.bn18(x)
        x = self.add9([temp_x, x])
        x = self.act18(x)
        temp_x = x
        x = self.conv2D20(x)
        x = self.bn19(x)
        x = self.act19(x)
        x = self.conv2D21(x)
        x = self.bn20(x)
        x = self.add10([temp_x, x])
        x = self.act20(x)
        temp_x = x
        x = self.conv2D22(x)
        x = self.bn21(x)
        x = self.act21(x)
        x = self.conv2D23(x)
        x = self.bn22(x)
        x = self.add11([temp_x, x])
        x = self.act22(x)
        temp_x = x
        x = self.conv2D24(x)
        x = self.bn23(x)
        x = self.act23(x)
        x = self.conv2D25(x)
        x = self.bn24(x)
        x = self.add12([temp_x, x])
        x = self.act24(x)
        temp_x = x
        x = self.conv2D26(x)
        x = self.bn25(x)
        x = self.act25(x)
        x = self.conv2D27(x)
        x = self.bn26(x)
        x = self.add13([temp_x, x])
        x = self.act26(x)
        temp_x = x

        # block 4
        x = self.conv2D28(x)
        x = self.bn27(x)
        x = self.act27(x)
        x = self.conv2D29(x)
        x = self.bn28(x)
        temp_x = self.sccov2D3(temp_x)
        x = self.add14([temp_x, x])
        x = self.act28(x)
        temp_x = x
        x = self.conv2D30(x)
        x = self.bn29(x)
        x = self.act29(x)
        x = self.conv2D31(x)
        x = self.bn30(x)
        x = self.add15([temp_x, x])
        x = self.act30(x)
        temp_x = x
        x = self.conv2D32(x)
        x = self.bn31(x)
        x = self.act31(x)
        x = self.conv2D33(x)
        x = self.bn32(x)
        x = self.add16([temp_x, x])
        x = self.act32(x)

        x = self.avep1(x)
        x = self.flatten1(x)

        # another branch which incorporates binary features
        # shallow binary feature
        # Entry block
        # 分成4个卷积层，sobel
        #x1 = self.rescale2(inputs)
        x1 = inputs

        x11 = self.conv2D32HB1(x1)
        # x11 = layers.Activation(activations.tanh(x11))
        x12 = self.conv2D32VB1(x1)
        # x12 = layers.Activation(activations.tanh(x12))
        x13 = self.conv2D32PB1(x1)
        # x13 = layers.Activation(activations.tanh(x13))
        x14 = self.conv2D32NB1(x1)
        #tf.print(x14[5, 0:32, 0:32, 20])
        # x14 = layers.Activation(activations.tanh(x14))

        # or to try average layer
        x2 = self.add1([x11, x12, x13, x14])
        #tf.print(x2[5, 0:32, 0:32, 20])
        #x2 = self.Batchnorm5(x2)

        x2 = self.maxPolling5(x2)
        x21 = self.conv2D64HB1(x2)
        #tf.print(x21[5, 0:16, 0:16, 50])
        # x21 = layers.Activation(activations.tanh(x21))
        x22 = self.conv2D64VB1(x2)
        # x22 = layers.Activation(activations.tanh(x22))
        x23 = self.conv2D64PB1(x2)
        # x23 = layers.Activation(activations.tanh(x23))
        x24 = self.conv2D64NB1(x2)
        # x24 = layers.Activation(activations.tanh(x24))

        x3 = self.add2([x21, x22, x23, x24])

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
        features = self.dense(features)
        outputs = self.classifier(features)
        return outputs
