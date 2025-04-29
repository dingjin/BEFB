def main_2layer_th8_cifar_vgg_binary(epochs = 200, batchsize = 50, repeat = 1, datetime='0601'):
    import tensorflow as tf
    import numpy as np
    import PIL
    from PIL import Image
    from tensorflow import keras
    from keras import layers
    from keras import utils
    #from unpickle import *
    import matplotlib.pyplot as plt
    from model_building_2layer_th8_vgg_binary import Vgg16binary
    import keras.backend as K

    """
    dict_batch1 = unpickle("cifar-10-python/data_batch_1")
    dict_batch2 = unpickle("cifar-10-python/data_batch_2")
    dict_batch3 = unpickle("cifar-10-python/data_batch_3")
    dict_batch4 = unpickle("cifar-10-python/data_batch_4")
    dict_batch5 = unpickle("cifar-10-python/data_batch_5")
    dict_test = unpickle("cifar-10-python/test_batch")
    """
    # im = Image.open("./G233-222.jpg")
    # im.show()

    # first, lets debug cifar-10 successfully
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

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
    # Print figure with 10 random images from each

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    x_train = x_train/255
    x_test = x_test/255

    inputs = keras.Input(shape=(32, 32, 3,))
    model_V = Vgg16binary()  # (inputs)
    # model_V.build(input_shape=(None, 32, 32, 3))
    # model_V.summary()
    # utils.plot_model(model_V, show_shapes=True)

    batch_size = batchsize
    num_epochs = epochs

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn = keras.losses.CategoricalCrossentropy()

    # Prepare the metrics.
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric = keras.metrics.CategoricalAccuracy()

    model_V.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    model_V.call(inputs)
    model_concat1 = keras.Model(inputs=model_V.layers[0].input, outputs=model_V.get_layer('conv2d_4').output)
    model_concat2 = keras.Model(inputs=model_V.layers[0].input, outputs=model_V.get_layer('dense').output)
    model_concat3 = keras.Model(inputs=model_V.layers[0].input, outputs=model_V.get_layer('dense_1').output)

    @tf.function
    def train_step(x, y, output_conv_dis, output_dense_dis, output_dense1_dis):
        with tf.GradientTape() as tape:
            probs = model_V(x, training=True)
            loss_value = loss_fn(y, probs)
            #output_conv_dis, output_dense_dis, output_dense1_dis = prototypeconformity(x, y)
            loss_value = loss_value + output_conv_dis + output_dense_dis + output_dense1_dis
        grads = tape.gradient(loss_value, model_V.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_V.trainable_weights))
        train_acc_metric.update_state(y, probs)
        return loss_value

    @tf.function
    def test_step(x, y):
        val_probs = model_V(x, training=False)
        val_acc_metric.update_state(y, val_probs)

    #@tf.function
    def prototypeconformity(x, y):
        # several layers output clusters

        output_conv = model_concat1(x, training=True)
        output_conv_dis = computing_distance_conv(output_conv, y)

        output_dense = model_concat2(x, training=True)
        output_dense_dis = computing_distance_dense(output_dense, y)

        output_dense1 = model_concat3(x, training=True)
        output_dense1_dis = computing_distance_dense(output_dense1, y)

        return output_conv_dis, output_dense_dis, output_dense1_dis

    #@tf.function
    def computing_distance_conv(inputfeatures, labels):
        # class center
        # every sample
        bts, width, height, channel = inputfeatures.shape

        #inputfeatures = tf.make_tensor_proto(inputfeatures)  # convert `tensor a` to a proto tensor
        #inputfeatures = tf.make_ndarray(inputfeatures)
        #classcenter = np.zeros((10, width, height, channel), dtype="float32")

        #classcenter = tf.constant(0, shape=(10, width, height, channel), dtype="float32")
        classnumber = tf.argmax(labels, 1)
        sort_order = tf.argsort(classnumber)
        sort_features = tf.gather(inputfeatures, sort_order)
        """
        i = 0
        while i < bts:
            classcenter[classnumber[i], 0:width, 0:height, 0:channel] += inputfeatures[i, 0:width, 0:height, 0:channel]
            i = i+1
        """

        samplesineachclass = tf.constant(0, shape=(1, 10))
        elements_equal_to_value = tf.equal(classnumber, 0)
        elements_equal_to_value = tf.expand_dims(elements_equal_to_value, 1)

        i = 1
        while i < 10:
            elements_equal_to_value_temp = tf.equal(classnumber, i)
            elements_equal_to_value_temp = tf.expand_dims(elements_equal_to_value_temp, 1)
            elements_equal_to_value = tf.concat([elements_equal_to_value, elements_equal_to_value_temp], 1)
            i = i + 1
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        samplesineachclass = tf.reduce_sum(as_ints, 0)
        samplesineachclass = tf.expand_dims(samplesineachclass, 0)

        mask = tf.cast(samplesineachclass, dtype=tf.bool)
        samplesineachclass = tf.boolean_mask(samplesineachclass, mask)
        samplesineachclass = tf.expand_dims(samplesineachclass, 0)

        # compute center

        #classcenter  = tf.reduce_mean(sort_features[0:samplesineachclass[0, samplesineachclass_realind[0,0]], 0:width, 0:height, 0:channel], 0)
        classcenter  = tf.reduce_mean(sort_features[0:samplesineachclass[0, 0]], 0)
        classcenter = tf.expand_dims(classcenter, 0)

        i = 1
        index_in_sort_features = samplesineachclass[0, 0]
        temp, numberofclass = samplesineachclass.shape

        while i < numberofclass:
            classcenter_temp = tf.reduce_mean(sort_features[index_in_sort_features:index_in_sort_features+samplesineachclass[0, i]], 0)
            index_in_sort_features = index_in_sort_features + samplesineachclass[0, i]
            classcenter_temp = tf.expand_dims(classcenter_temp, 0)

            #tf.autograph.experimental.set_loop_options(shape_invariants=[(classcenter, tf.TensorShape([None]))])
            classcenter = tf.concat([classcenter, classcenter_temp], 0)
            i = i + 1

        # compute distance
        distance = 0
        i = 0
        # compute the distance feature to own center
        count_index = 0


        own_feature_center_sub = tf.repeat(
            tf.expand_dims(classcenter[i],0),
            samplesineachclass[0, count_index % numberofclass], axis=0)


        count_index = count_index + 1
        i = i + 1
        temp, numberofclass = samplesineachclass.shape
        while i < numberofclass:
            own_feature_center_sub_temp = tf.repeat(
                tf.expand_dims(classcenter[i % numberofclass],0),
                samplesineachclass[0, count_index % numberofclass], axis=0)
            #tf.autograph.experimental.set_loop_options(shape_invariants=[(own_feature_center_sub, tf.TensorShape([None]))])

            own_feature_center_sub = tf.concat(
                [own_feature_center_sub, own_feature_center_sub_temp], 0)
            i = i + 1
            count_index = count_index + 1

        distance += tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(sort_features-own_feature_center_sub), [1, 2, 3])))
        # compute distance feature to other class center
        distance_temp = 0
        i = 1
        while i < numberofclass:
            count_index = 0
            j = i
            other_feature_center_sub = tf.repeat(
                tf.expand_dims(classcenter[j % numberofclass], 0),
                samplesineachclass[0, count_index % numberofclass], axis=0)
            j = j + 1
            temp = j
            count_index = count_index + 1
            while j < temp+numberofclass-1:
                other_feature_center_sub_temp = tf.repeat(tf.expand_dims(classcenter[j%numberofclass], 0), samplesineachclass[0, count_index % numberofclass], axis = 0)
                other_feature_center_sub = tf.concat([other_feature_center_sub, other_feature_center_sub_temp], 0)
                j = j + 1
                count_index = count_index + 1
            i = i + 1

            distance_temp += tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(tf.square(sort_features - other_feature_center_sub), [1, 2, 3])))

        distance = distance - distance_temp/(numberofclass-1)

        # compute center to center
        distance_temp = 0
        i = 0
        classcenter_temp = classcenter
        samplesineachclass = tf.cast(samplesineachclass, tf.float32)
        while i < numberofclass:
            classcenter_temp = tf.roll(classcenter_temp, [-1], [0])
            temp = tf.reduce_sum(tf.square(classcenter - classcenter_temp), [1, 2, 3])
            temp = tf.expand_dims(temp, 0)
            #print(temp)
            #print(samplesineachclass)

            temp = tf.multiply(samplesineachclass, temp)
            distance_temp = distance_temp + tf.reduce_sum(tf.sqrt(temp))

            i = i + 1
        distance = distance - distance_temp/(numberofclass-1)

        return distance

    #@tf.function
    def computing_distance_dense(inputfeatures, labels):
        # class center
        # every sample
        bts, dimensions = inputfeatures.shape

        # inputfeatures = tf.make_tensor_proto(inputfeatures)  # convert `tensor a` to a proto tensor
        # inputfeatures = tf.make_ndarray(inputfeatures)
        # classcenter = np.zeros((10, width, height, channel), dtype="float32")

        # classcenter = tf.constant(0, shape=(10, width, height, channel), dtype="float32")
        classnumber = tf.argmax(labels, 1)
        sort_order = tf.argsort(classnumber)
        sort_features = tf.gather(inputfeatures, sort_order)
        """
        i = 0
        while i < bts:
            classcenter[classnumber[i], 0:width, 0:height, 0:channel] += inputfeatures[i, 0:width, 0:height, 0:channel]
            i = i+1
        """


        elements_equal_to_value = tf.equal(classnumber, 0)
        elements_equal_to_value = tf.expand_dims(elements_equal_to_value, 1)

        i = 1
        while i < 10:
            elements_equal_to_value_temp = tf.equal(classnumber, i)
            elements_equal_to_value_temp = tf.expand_dims(elements_equal_to_value_temp, 1)
            elements_equal_to_value = tf.concat([elements_equal_to_value, elements_equal_to_value_temp], 1)
            i = i + 1
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        samplesineachclass = tf.reduce_sum(as_ints, 0)
        samplesineachclass = tf.expand_dims(samplesineachclass, 0)

        mask = tf.cast(samplesineachclass, dtype=tf.bool)
        samplesineachclass = tf.boolean_mask(samplesineachclass, mask)
        samplesineachclass = tf.expand_dims(samplesineachclass, 0)

        temp, numberofclass = samplesineachclass.shape

        # compute center

        # classcenter  = tf.reduce_mean(sort_features[0:samplesineachclass[0, samplesineachclass_realind[0,0]], 0:width, 0:height, 0:channel], 0)
        classcenter = tf.reduce_mean(sort_features[0:samplesineachclass[0, 0]], 0)
        classcenter = tf.expand_dims(classcenter, 0)

        i = 1
        index_in_sort_features = samplesineachclass[0, 0]

        while i < numberofclass:
            classcenter_temp = tf.reduce_mean(sort_features[
                                              index_in_sort_features:index_in_sort_features + samplesineachclass[
                                                  0, i]], 0)
            index_in_sort_features = index_in_sort_features + samplesineachclass[0, i]
            classcenter_temp = tf.expand_dims(classcenter_temp, 0)

            #tf.autograph.experimental.set_loop_options(shape_invariants=[(classcenter, tf.TensorShape([None]))])
            classcenter = tf.concat([classcenter, classcenter_temp], 0)
            i = i + 1

        # compute distance
        distance = 0
        i = 0
        # compute the distance feature to own center
        count_index = 0

        own_feature_center_sub = tf.repeat(
            tf.expand_dims(classcenter[i], 0),
            samplesineachclass[0, count_index % numberofclass], axis=0)

        count_index = count_index + 1
        i = i + 1
        temp, numberofclass = samplesineachclass.shape
        while i < numberofclass:
            own_feature_center_sub_temp = tf.repeat(
                tf.expand_dims(classcenter[i % numberofclass], 0),
                samplesineachclass[0, count_index % numberofclass], axis=0)
            #tf.autograph.experimental.set_loop_options(shape_invariants=[(own_feature_center_sub, tf.TensorShape([None]))])

            own_feature_center_sub = tf.concat(
                [own_feature_center_sub, own_feature_center_sub_temp], 0)
            i = i + 1
            count_index = count_index + 1
        distance += tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(sort_features - own_feature_center_sub), [1])))
        # compute distance feature to other class center
        distance_temp = 0
        i = 1
        while i < numberofclass:
            count_index = 0
            j = i
            other_feature_center_sub = tf.repeat(
                tf.expand_dims(classcenter[j % numberofclass], 0),
                samplesineachclass[0, count_index % numberofclass], axis=0)
            j = j + 1
            temp = j
            count_index = count_index + 1
            while j < temp + numberofclass - 1:
                other_feature_center_sub_temp = tf.repeat(tf.expand_dims(classcenter[j % numberofclass], 0),
                                                          samplesineachclass[0, count_index % numberofclass], axis=0)
                other_feature_center_sub = tf.concat([other_feature_center_sub, other_feature_center_sub_temp], 0)
                j = j + 1
                count_index = count_index + 1
            i = i + 1
            distance_temp += tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(tf.square(sort_features - other_feature_center_sub), [1])))

        distance = distance - distance_temp / (numberofclass - 1)

        # compute center to center
        distance_temp = 0
        i = 0
        classcenter_temp = classcenter
        samplesineachclass = tf.cast(samplesineachclass, tf.float32)
        while i < numberofclass:
            classcenter_temp = tf.roll(classcenter_temp, [-1], [0])
            temp = tf.reduce_sum(tf.square(classcenter - classcenter_temp), [1])
            temp = tf.expand_dims(temp, 0)
            #print(temp)
            #print(samplesineachclass)

            temp = tf.multiply(samplesineachclass, temp)
            distance_temp = distance_temp + tf.reduce_sum(tf.sqrt(temp))
            i = i + 1
        distance = distance - distance_temp / (numberofclass - 1)

        return distance



    """
    #@tf.function
    def computing_distance_dense(inputfeatures, labels):
        # class center
        # every sample
        bts, dimensions = inputfeatures.shape
        #inputfeatures = tf.make_tensor_proto(inputfeatures)  # convert `tensor a` to a proto tensor
        #inputfeatures = tf.make_ndarray(inputfeatures)
        classcenter = np.zeros((10, dimensions), dtype="float32")
        #classcenter = tf.constant(0, shape=(10, dimensions))
        classnumber = tf.argmax(labels, 1)
        i = 0
        while i < bts:
            classcenter[classnumber[i], 0:dimensions] += inputfeatures[i, 0:dimensions]
            i = i+1
        samplesineachclass = np.zeros((1, 10), dtype="int32")
        # samplesineachclass = tf.constant(0, shape=(1, 10))
        i = 0
        while i < 10:
            elements_equal_to_value = tf.equal(classnumber, i)
            as_ints = tf.cast(elements_equal_to_value, tf.int32)
            samplesineachclass[0, i] = tf.reduce_sum(as_ints)
            i = i + 1
        i = 0
        while i < 10:
            if samplesineachclass[0, i] > 0:
                classcenter[i, 0:dimensions] /= samplesineachclass[0, i]
            i = i + 1
        # compute distance
        distance = 0
        i = 0
        while i < bts:
            distance += tf.reduce_sum(tf.square(inputfeatures[i, 0:dimensions] - classcenter[classnumber[i], 0:dimensions]))
            j = 0
            distance_temp = 0
            while j < 10:
                if classnumber[i] != j:
                    distance_temp += tf.reduce_sum(tf.square(inputfeatures[i, 0:dimensions] - classcenter[j, 0:dimensions]))
                    distance_temp += tf.reduce_sum(tf.square(classcenter[classnumber[i], 0:dimensions] - classcenter[j, 0:dimensions]))
                j = j + 1
            distance -= distance_temp/(10-1)
            i = i + 1
        return distance
    """
    import time
    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train_binary_ndarray[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train_binary_ndarray[:-10000]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batchsize)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batchsize)

    for epoch in range(num_epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            output_conv_dis, output_dense_dis, output_dense1_dis = prototypeconformity(x_batch_train, y_batch_train)
            loss_value = train_step(x_batch_train, y_batch_train, output_conv_dis, output_dense_dis, output_dense1_dis)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                train_acc = train_acc_metric.result()
                print("Training acc (for one batch) at step %d: %.4f" % (step, float(train_acc)))
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

    score = model_V.evaluate(x_test, y_test_binary_ndarray, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    # model_V.save('VGGBinaryModel0207-float')
    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/PCL/weiVGGBinaryModel'
    pathstring += datetime
    pathstring += '-cifar10-2layer-th8-repeat'
    pathstring += str(repeat)
    pathstring += '/my_checkpoint'
    model_V.save_weights(pathstring)

    return
import sys
if __name__ == '__main__':
    argv1 = 100#int(sys.argv[1])
    argv2 = 100#int(sys.argv[2])
    argv3 = 1#int(sys.argv[3])
    argv4 = '0601'#sys.argv[4]
    main_2layer_th8_cifar_vgg_binary(epochs=argv1, batchsize=argv2, repeat = argv3, datetime=argv4)


'''
model_V.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
'''

