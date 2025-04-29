# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of adversarial Keras trainer on classifying MNIST images.
USAGE:
  python adv_keras_cnn_mnist.py
See http://yann.lecun.com/exdb/mnist/ for the description of the MNIST dataset.
This example demonstrates how to train a Keras model with adversarial
regularization. The base model demonstrated in this example is a convolutional
neural network built with Keras functional APIs, and users are encouraged to
modify the `build_base_model()` function to try other types of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import attr
import neural_structured_learning as nsl
import tensorflow as tf
import matplotlib.pyplot as plt
#from model_building_VGG import Vgg16
import numpy as np
from keras.utils import get_custom_objects
from keras.constraints import Constraint
import keras.backend as K
from keras import layers
from keras import activations
from tensorflow import keras
# sobel horizon weight constraint and fix
import global_variable
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
def threshold2binary(x, t=0.4):
    def _threshold2binary_grad(grad):
        #tf.print(global_variable.global_gradient_flag)
        return global_variable.global_gradient_flag * grad
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

    return x, _threshold2binary_grad


get_custom_objects().update({'threshold2binary': layers.Activation(threshold2binary)})


FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', None, 'Number of epochs to train.')
flags.DEFINE_integer('steps_per_epoch', None,
                     'Number of steps in each training epoch.')
flags.DEFINE_integer('eval_steps', None, 'Number of steps to evaluate.')
flags.DEFINE_float('adv_step_size', None,
                   'Step size for generating adversarial examples.')

FEATURE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'


@attr.s
class HParams(object):
    """Hyper-parameters for training the model."""
    # model architecture parameters
    input_shape = attr.ib(default=(32, 32, 3))
    conv_filters = attr.ib(default=[64, 128, 256, 512])
    kernel_size = attr.ib(default=(3, 3))
    pool_size = attr.ib(default=(2, 2))
    dense_units = attr.ib(default=[4096, 4096])
    num_classes = attr.ib(default=10)
    # adversarial parameters
    adv_multiplier = attr.ib(default=0.2)
    adv_step_size = attr.ib(default=2/255)
    adv_grad_norm = attr.ib(default='infinity')

    clip_value_max = attr.ib(default=None)
    clip_value_min = attr.ib(default=None)
    feature_mask = attr.ib(default=None)
    pgd_epsilon = attr.ib(default=8/255)
    pgd_iterations = attr.ib(default=8)
    random_init = attr.ib(default=False)


    # training parameters
    batch_size = attr.ib(default=50)
    buffer_size = attr.ib(default=10000)
    epochs = attr.ib(default=200)
    steps_per_epoch = attr.ib(default=None)
    eval_steps = attr.ib(default=None)


def get_hparams():
    """Returns the hyperparameters with defaults overwritten by flags."""
    hparams = HParams()
    if FLAGS.epochs:
        hparams.epochs = FLAGS.epochs
    if FLAGS.adv_step_size:
        hparams.adv_step_size = FLAGS.adv_step_size
    if FLAGS.steps_per_epoch:
        hparams.steps_per_epoch = FLAGS.steps_per_epoch
    if FLAGS.eval_steps:
        hparams.eval_steps = FLAGS.eval_steps
    return hparams


def prepare_datasets(hparams):
    """Downloads the MNIST dataset and converts to `tf.data.Dataset` format."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_train_binary = tf.zeros([50000, 10], dtype=tf.int32)
    y_test_binary = tf.zeros([10000, 10], dtype=tf.int32)
    # Converting Tensor to TensorProto
    proto = tf.make_tensor_proto(y_train_binary)
    # Generating numpy array
    y_train_binary_ndarray = tf.make_ndarray(proto)

    # Converting Tensor to TensorProto
    proto = tf.make_tensor_proto(y_test_binary)
    # Generating numpy array
    y_test_binary_ndarray = tf.make_ndarray(proto)

    for i in range(50000):
        y_train_binary_ndarray[i, y_train[i, 0]] = 1
    for i in range(10000):
        y_test_binary_ndarray[i, y_test[i, 0]] = 1

    def make_dataset(x, y, shuffle=False):
        x = x.reshape((-1, 32, 32, 3)).astype('float32') / 255.0
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            dataset = dataset.shuffle(hparams.buffer_size)
        return dataset.batch(hparams.batch_size)

    return make_dataset(x_train, y_train_binary_ndarray, True), make_dataset(x_test, y_test_binary_ndarray)


def convert_to_adversarial_training_dataset(dataset):
    def to_dict(x, y):
        return {FEATURE_INPUT_NAME: x, LABEL_INPUT_NAME: y}

    return dataset.map(to_dict)


def build_base_model(hparams):
    """Builds a model according to the architecture defined in `hparams`."""

    inputs = tf.keras.Input(
        shape=hparams.input_shape, dtype=tf.float32, name=FEATURE_INPUT_NAME)
    x = inputs
    x = layers.Conv2D(64, 7, strides=1, padding="same", activation="relu")(x)
    x_temp = x
    # block 1  3x3 64 2  x3
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x_temp, x])
    x = layers.Activation(activations.relu)(x)

    x_temp = x
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x_temp, x])
    x = layers.Activation(activations.relu)(x)
    x_temp = x
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x_temp, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x

    # block 2
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    temp_x = layers.Conv2D(128, 1, strides=2, padding="same")(temp_x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x

    # block 3
    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    temp_x = layers.Conv2D(256, 1, strides=2, padding="same")(temp_x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(256, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x

    # block 4
    x = layers.Conv2D(512, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(512, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    temp_x = layers.Conv2D(512, 1, strides=2, padding="same")(temp_x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(512, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(512, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)
    temp_x = x
    x = layers.Conv2D(512, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(512, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([temp_x, x])
    x = layers.Activation(activations.relu)(x)

    x = layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
    x = layers.Flatten()(x)

    x1 = inputs

    x11 = layers.Conv2D(32, 3, strides=1, padding="same",
                                         kernel_constraint=HBetweenAndFix(),
                                         activation='relu')(x1)

    x12 = layers.Conv2D(32, 3, strides=1, padding="same",
                                         kernel_constraint=VBetweenAndFix(),
                                         activation='relu')(x1)

    x13 = layers.Conv2D(32, 3, strides=1, padding="same",
                                         kernel_constraint=PDBetweenAndFix(),
                                         activation='relu')(x1)

    x14 = layers.Conv2D(32, 3, strides=1, padding="same",
                                         kernel_constraint=NDBetweenAndFix(),
                                         activation='relu')(x1)

    x2 = layers.Add()([x11, x12, x13, x14])

    x2 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x2)
    x21 = layers.Conv2D(64, 3, strides=1, padding="same",
                                         kernel_constraint=HBetweenAndFix(),
                                         activation='relu')(x2)

    x22 = layers.Conv2D(64, 3, strides=1, padding="same",
                                         kernel_constraint=VBetweenAndFix(),
                                         activation='relu')(x2)

    x23 = layers.Conv2D(64, 3, strides=1, padding="same",
                                         kernel_constraint=PDBetweenAndFix(),
                                         activation='relu')(x2)

    x24 = layers.Conv2D(64, 3, strides=1, padding="same",
                                         kernel_constraint=NDBetweenAndFix(),
                                         activation='relu')(x2)


    x3 = layers.Add()([x21, x22, x23, x24])


    x3 = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x3)

    x4 = layers.Activation(threshold2binary)(x3)

    x4 = layers.Flatten()(x4)

    x = layers.Concatenate()([x, x4])

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10)(x)
    pred = layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=pred)

    #model = Vgg16()
    return model


def apply_adversarial_regularization(model, hparams):
    adv_config = nsl.configs.make_adv_reg_config(
        multiplier=hparams.adv_multiplier,
        adv_step_size=hparams.adv_step_size,
        adv_grad_norm=hparams.adv_grad_norm)
    return nsl.keras.AdversarialRegularization(
        model, label_keys=[LABEL_INPUT_NAME], adv_config=adv_config)


def build_adv_model(hparams):
    """Builds an adversarial-regularized model from parameters in `hparams`."""
    base_model = build_base_model(hparams)
    return apply_adversarial_regularization(base_model, hparams)


def train_and_evaluate(model, hparams, train_dataset, test_dataset):
    """Trains the model and returns the evaluation result."""
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(
        train_dataset,
        epochs=hparams.epochs,
        steps_per_epoch=hparams.steps_per_epoch)

    eval_result = model.evaluate(test_dataset, steps=hparams.eval_steps)
    return list(zip(model.metrics_names, eval_result))


def evaluate_robustness(model_to_attack, dataset, models, hparams):
    """Evaluates the robustness of `models` with adversarially-perturbed input.
    Args:
      model_to_attack: `tf.keras.Model`. Perturbations will be generated based
        on this model's weights.
      dataset: Dataset to be perturbed.
      models: Dictionary of model names and `tf.keras.Model` to be evaluated.
      hparams: Hyper-parameters for generating adversarial examples.
    Returns:
      A dictionary of model names and accuracy of the model on adversarially
      perturbed input, i.e. robustness.
    """
    if not isinstance(model_to_attack, nsl.keras.AdversarialRegularization):
        # Enables AdversarialRegularization-specific API for the model_to_attack.
        # This won't change the model's weights.
        model_to_attack = apply_adversarial_regularization(model_to_attack, hparams)
        model_to_attack.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    metrics = {
        name: tf.keras.metrics.CategoricalAccuracy()
        for name in models.keys()
    }

    if hparams.eval_steps:
        dataset = dataset.take(hparams.eval_steps)
    # When running on accelerators, looping over the dataset inside a tf.function
    # may be much faster.
    perturbed_images, labels, predictions = [], [], []
    for batch in dataset:
        adv_batch = model_to_attack.perturb_on_batch(batch)
        # Clips the perturbed values to 0~1, the same as normalized values after
        # preprocessing.
        adv_batch[FEATURE_INPUT_NAME] = tf.clip_by_value(
            adv_batch[FEATURE_INPUT_NAME], 0.0, 1.0)
        y_true = adv_batch.pop(LABEL_INPUT_NAME)

        perturbed_images.append(adv_batch[FEATURE_INPUT_NAME].numpy())
        labels.append(y_true.numpy())
        predictions.append({})
        for name, model in models.items():
            y_pred = model(adv_batch)
            metrics[name](y_true, y_pred)
            predictions[-1][name] = tf.argmax(y_pred, axis=-1).numpy()

    return {name: metric.result().numpy() for name, metric in metrics.items()}, perturbed_images, labels, predictions


def main(argv):
    del argv  # Unused.



    hparams = get_hparams()

    train_dataset, test_dataset = prepare_datasets(hparams)


    adv_model = build_adv_model(hparams)
    adv_train_dataset = convert_to_adversarial_training_dataset(train_dataset)
    adv_test_dataset = convert_to_adversarial_training_dataset(test_dataset)


    adv_result = train_and_evaluate(adv_model, hparams, adv_train_dataset, adv_test_dataset)
    for metric_name, result in adv_result:
        print('Eval %s for adversarial model: %s' % (metric_name, result))
    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/ATtraing/weiResNetbinaryModel'
    pathstring += '0601'
    pathstring += '-2layer-th4-AT-repeat'
    pathstring += str(1)
    pathstring += '/my_checkpoint'
    adv_model.base_model.save_weights(pathstring)

    models = {

        # Takes the base model from adv_model so that input format is the same.
        'adv-regularized': adv_model.base_model,
    }

    adv_accuracy, perturbed_images, labels, predictions = evaluate_robustness(adv_model, adv_test_dataset, models,
                                       hparams)
    print('----- Adversarial attack on adv model -----')
    for name, accuracy in adv_accuracy.items():
        print('%s model accuracy: %f' % (name, accuracy))
    score = np.array([accuracy], dtype=np.float32)

    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/ATtraing/'
    namestring = 'weiResNetbinaryModel'
    namestring += '0601'
    namestring += '-2layer-th4-AT-repeat'

    headerstring = namestring
    namestring += '-pgd-8-2-8'

    namestring += '.txt'

    with open(pathstring + namestring, 'a') as f:
        np.savetxt(f, score, delimiter=",", header=headerstring, fmt='%.4f', newline=' ')
        f.write("\n")

if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    app.run(main)
