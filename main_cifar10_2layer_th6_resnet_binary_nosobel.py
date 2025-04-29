def main_2layer_th6_cifar_resnet_binary(epochs = 200, batchsize = 50, repeat = 1, datetime='0601'):
    import tensorflow as tf
    import numpy as np
    import PIL
    from PIL import Image
    from tensorflow import keras
    from keras import layers
    from keras import utils
    #from unpickle import *
    import matplotlib.pyplot as plt
    from model_building_2layer_th6_resnet_binary_nosobel import ResNet34binary
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

    """
    fig = plt.figure(figsize=(8,3))
    for i in range(10):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:]==i)[0]
        features_idx = x_train[idx,::]
        img_num = np.random.randint(features_idx.shape[0])
        im = np.transpose(features_idx[img_num,::],(1,0,2))
        #ax.set_title(class_names[i])
        plt.imshow(im)
    plt.show()
    """
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    x_test = x_test/255
    x_train = x_train/255

    inputs = keras.Input(shape=(32, 32, 3,))
    model_V = ResNet34binary()  # (inputs)
    # model_V.build(input_shape=(None, 32, 32, 3))
    # model_V.summary()
    # utils.plot_model(model_V, show_shapes=True)

    batch_size = batchsize
    num_epochs = epochs

    model_V.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        # monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model_V.fit(
        x=x_train,
        y=y_train_binary_ndarray,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    # Plots for training and testing process: loss and accuracy
    """
    plt.figure(0)
    plt.plot(history.history['accuracy'],'r')
    plt.plot(history.history['val_accuracy'],'g')
    plt.xticks(np.arange(0, num_epochs+1, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])


    plt.figure(1)
    plt.plot(history.history['loss'],'r')
    plt.plot(history.history['val_loss'],'g')
    plt.xticks(np.arange(0, num_epochs+1, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])

    plt.show()
    """

    score = model_V.evaluate(x_test, y_test_binary_ndarray, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    # model_V.save('VGGBinaryModel0207-float')
    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/dataset-cifar10-ablation/weiResNetBinaryModel'
    pathstring += datetime
    pathstring += '-2layer-th6-nosobel-repeat'
    pathstring += str(repeat)
    pathstring += '/my_checkpoint'
    model_V.save_weights(pathstring)

    return
import sys
if __name__ == '__main__':
    argv1 = int(sys.argv[1])
    argv2 = int(sys.argv[2])
    argv3 = int(sys.argv[3])
    main_2layer_th6_cifar_resnet_binary(epochs=argv1, batchsize=argv2, repeat = argv3, datetime=sys.argv[4])



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

