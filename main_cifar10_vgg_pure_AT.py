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
from tensorflow import keras
# sobel horizon weight constraint and fix


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
    epochs = attr.ib(default=100)
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
    for filter_idx, num_filters in enumerate(hparams.conv_filters):
        x = tf.keras.layers.Conv2D(
            num_filters, hparams.kernel_size, strides=1, padding="same", activation='relu')(
            x)
        if filter_idx < len(hparams.conv_filters) - 1:
            # max pooling between convolutional layers
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(hparams.pool_size, strides=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(hparams.pool_size, strides=None, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)

    for num_hidden_units in hparams.dense_units:
        x = tf.keras.layers.Dense(num_hidden_units, activation='sigmoid')(x)
        x = layers.Dropout(0.2)(x)
    pred = tf.keras.layers.Dense(hparams.num_classes, activation='softmax')(x)
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
    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/ATtraing/weiVGGpureModel'
    pathstring += '0601'
    pathstring += '-AT-repeat'
    pathstring += str(2)
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
    namestring = 'weiVGGpureModel'
    namestring += '0601'
    namestring += '-AT-repeat'

    headerstring = namestring
    namestring += '-pgd-8-2-8'

    namestring += '.txt'

    with open(pathstring + namestring, 'a') as f:
        np.savetxt(f, score, delimiter=",", header=headerstring, fmt='%.4f', newline=' ')
        f.write("\n")

if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    app.run(main)
