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

class ResNet34(keras.Model):
    def __init__(self, name="ResNet34", **kwargs):
        super(ResNet34, self).__init__()
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

        # fully connected layers
        self.dropout1 = layers.Dropout(0.2)
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

        x = self.dropout1(x)
        x = self.dense(x)
        output = self.classifier(x)
        return output

