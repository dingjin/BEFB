def comp_cifar10_vgg_pure_PCL(numberofcommon=0, repeat=1, datetime='0601', epsilons=8, stepsize=1, steps=8, numberofcommoncw=100):
    import tensorflow as tf
    import numpy as np
    import PIL
    from PIL import Image
    from tensorflow import keras
    from keras import layers
    from keras import utils
    # from unpickle import *
    import matplotlib.pyplot as plt
    import keras.backend as K
    import matplotlib as mpl
    # from model_building_shallowBinary_fixed import VGG16_shallowB

    # from model_building_shallowBinary import VGG16_shallowB
    # from model_building_shallowBinary_relu import VGG16_shallowB
    # from model_building import Vgg16binary
    from model_building_VGG import Vgg16
    # from model_building_3layer_th8_deep import Vgg16binary




    # resotre model
    # model_binary = tf.keras.models.load_model('VGGShallowBModel0207-16x16x64-float')#VGGpureModel0207-float VGGShallowBModel0207-16x16x64-float

    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/PCL/weiVGGpureModel'
    pathstring += datetime
    pathstring += '-cifar10-repeat'
    pathstring += str(repeat)
    pathstring += '/my_checkpoint'
    model_binary = Vgg16()
    model_binary.load_weights(pathstring)
    #model_binary.training = False
    #model_binary.summary()

    model_binary1 = model_binary
    mpl.rcParams['figure.figsize'] = (2, 2)
    # mpl.rcParams['axes.grid'] = False

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    assert x_test.shape == (10000, 32, 32, 3)

    assert y_test.shape == (10000, 1)

    y_test_binary = tf.zeros([10000, 10], dtype=tf.int32)

    # Converting Tensor to TensorProto
    proto = tf.make_tensor_proto(y_test_binary)
    # Generating numpy array
    y_test_binary_ndarray = tf.make_ndarray(proto)

    for i in range(10000):
        y_test_binary_ndarray[i, y_test[i, 0]] = 1

    x_test = tf.cast(x_test, tf.float32)
    y_test_binary_ndarray = tf.cast(y_test_binary_ndarray, tf.float32)
    x_test = x_test/255.0

    model_binary.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_binary1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = model_binary.evaluate(x_test, y_test_binary_ndarray, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    score = model_binary1.evaluate(x_test, y_test_binary_ndarray, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    class_name = tf.constant(
        [['airplane'], ["automobile"], ["bird"], ["cat"], ["deer"], ["dog"], ["frog"], ["horse"], ["ship"], ["truck"]])

    image_probs = model_binary.predict(x_test)
    image_pre_class = tf.argmax(image_probs, 1)
    image_pre_class = tf.cast(image_pre_class, tf.uint8)
    image_pre_class = tf.expand_dims(image_pre_class, 1)
    cond = tf.equal(image_pre_class, y_test)
    # return opposite
    cond_opp = tf.logical_not(cond)
    pred_false_ind = tf.where(cond_opp)
    pre_true_ind = tf.where(cond)

    image_probs1 = model_binary1.predict(x_test)
    image_pre_class1 = tf.argmax(image_probs1, 1)
    image_pre_class1 = tf.cast(image_pre_class1, tf.uint8)
    image_pre_class1 = tf.expand_dims(image_pre_class1, 1)
    cond1 = tf.equal(image_pre_class1, y_test)
    # return opposite
    cond_opp1 = tf.logical_not(cond1)
    pred_false_ind1 = tf.where(cond_opp1)
    pre_true_ind1 = tf.where(cond1)
    number, temp = pre_true_ind.shape
    number1, temp = pre_true_ind1.shape
    common_index = tf.constant([-1], dtype=tf.int64)
    common_index = tf.expand_dims(common_index, 0)
    if (tf.less(number, number1)):
        for i in range(number):
            cond = tf.equal(pre_true_ind1[:, 0], pre_true_ind[i, 0])
            indexnumber = tf.where(cond)
            is_empty = tf.equal(tf.size(indexnumber), 0)
            if not is_empty:
                temp = pre_true_ind[i, 0]
                temp = tf.expand_dims(temp, 0)
                temp = tf.expand_dims(temp, 0)
                common_index = tf.concat([common_index, temp], 0)
    else:
        for i in range(number1):
            cond = tf.equal(pre_true_ind[:, 0], pre_true_ind1[i, 0])
            indexnumber = tf.where(cond)
            is_empty = tf.equal(tf.size(indexnumber), 0)
            if not is_empty:
                temp = pre_true_ind1[i, 0]
                temp = tf.expand_dims(temp, 0)
                temp = tf.expand_dims(temp, 0)
                common_index = tf.concat([common_index, temp], 0)

    common_index = common_index[1:, :]
    numberofcommonindex, temp = common_index.shape
    if numberofcommon != 0:
        numberofcommonindex = numberofcommon
    if numberofcommoncw == 0:
        numberofcommoncw = numberofcommonindex
    """
    x_test = tf.cast(x_test, tf.float32)
    plt.figure()
    plt.imshow(x_test[1]/255)  # To change [-1, 1] to [0,1]
    plt.title("{} : {:.2f}% Confidence".format(class_name[int(image_pre_class[1])], float(image_probs[1, image_pre_class[1]]*100)))
    plt.show()
    """

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model_binary(input_image)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        # g_abs_max = tf.reduce_max(tf.abs(gradient))
        # he_abs_max = tf.reduce_max(tf.abs(he_direction_reshape_expand))
        # g_normalized = tf.multiply(gradient, 1.0 / g_abs_max)
        return signed_grad  # signed_grad

    # plt.imshow(perturbations[0]/255.0)  # To change [-1, 1] to [0,1]

    eps = epsilons


    # x_test = tf.convert_to_tensor(x_test)
    # y_test_binary_ndarray = tf.convert_to_tensor(y_test_binary_ndarray)
    temp = x_test[common_index[0, 0].numpy()]
    temp = tf.expand_dims(temp, 0)
    temp_y = y_test_binary_ndarray[common_index[0, 0].numpy()]
    temp_y = tf.expand_dims(temp_y, 0)
    y_test_binary_ndarray_common = temp_y
    y_test_common = y_test[common_index[0, 0].numpy()]
    y_test_common = tf.expand_dims(y_test_common, 0)
    perturbations = create_adversarial_pattern(temp, temp_y)
    # tf.keras.utils.save_img("oriimage.jpg", temp[0,:,:,:], data_format="channels_last")
    adv_x_0 = temp + eps * perturbations/255.0
    # tf.keras.utils.save_img("advimage.jpg", adv_x_0[0,:,:,:], data_format="channels_last")
    i = tf.constant(1)
    count = tf.constant(0)
    zerogra = 0
    while (tf.less(i, numberofcommonindex)):
        temp = x_test[common_index[i, 0]]
        temp = tf.expand_dims(temp, 0)
        temp_y = y_test_binary_ndarray[common_index[i, 0]]
        temp_y = tf.expand_dims(temp_y, 0)
        y_test_binary_ndarray_common = tf.concat([y_test_binary_ndarray_common, temp_y], 0)
        temp_y_common = y_test[common_index[i, 0].numpy()]
        temp_y_common = tf.expand_dims(temp_y_common, 0)
        y_test_common = tf.concat([y_test_common, temp_y_common], 0)
        perturbations = create_adversarial_pattern(temp, temp_y)
        if tf.reduce_max(perturbations) == tf.reduce_min(perturbations):
            assert tf.reduce_max(perturbations) == 0
            print(perturbations)
            zerogra = zerogra + 1
        count = tf.add(count, 1)
        adv_x = temp + eps * perturbations/255.0
        adv_x_0 = tf.concat([adv_x_0, adv_x], 0)
        i = tf.add(i, 1)

    adv_x_0 = tf.clip_by_value(adv_x_0, 0.0, 1.0)
    score = model_binary.evaluate(adv_x_0, y_test_binary_ndarray_common, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print(numberofcommonindex)
    print(zerogra)
    score[1] = (numberofcommonindex * score[1] - zerogra) / (numberofcommonindex - zerogra)

    # save loss and accuracy, append mode 2023-6-4
    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/PCL/'
    namestring = 'weiVGGpureModel'
    namestring += datetime
    namestring += '-cifar10-repeat'

    headerstring = namestring
    namestring += '-fgsm'
    namestring += str(epsilons)
    namestring += '.txt'

    with open(pathstring+namestring, 'a') as f:
        np.savetxt(f, score, delimiter=",", header=headerstring, fmt='%.4f', newline=' ')
        f.write("\n")


    # pgd

   # 先跑一次

    temp = x_test[common_index[0, 0].numpy()]
    temp = tf.expand_dims(temp, 0)
    temp_y = y_test_binary_ndarray[common_index[0, 0].numpy()]
    temp_y = tf.expand_dims(temp_y, 0)
    y_test_binary_ndarray_common = temp_y
    y_test_common = y_test[common_index[0, 0].numpy()]
    y_test_common = tf.expand_dims(y_test_common, 0)
    perturbations = create_adversarial_pattern(temp, temp_y)
    totalchangge_0 = stepsize * perturbations/255.0
    # tf.keras.utils.save_img("oriimage.jpg", temp[0,:,:,:], data_format="channels_last")
    adv_x_0 = temp + totalchangge_0
    # tf.keras.utils.save_img("advimage.jpg", adv_x_0[0,:,:,:], data_format="channels_last")
    i = tf.constant(1)

    while (tf.less(i, numberofcommonindex)):
        temp = x_test[common_index[i, 0]]
        temp = tf.expand_dims(temp, 0)
        temp_y = y_test_binary_ndarray[common_index[i, 0]]
        temp_y = tf.expand_dims(temp_y, 0)
        y_test_binary_ndarray_common = tf.concat([y_test_binary_ndarray_common, temp_y], 0)
        temp_y_common = y_test[common_index[i, 0].numpy()]
        temp_y_common = tf.expand_dims(temp_y_common, 0)
        y_test_common = tf.concat([y_test_common, temp_y_common], 0)
        perturbations = create_adversarial_pattern(temp, temp_y)
        if tf.reduce_max(perturbations) == tf.reduce_min(perturbations):
            assert tf.reduce_max(perturbations) == 0
        totalchangge = stepsize * perturbations/255.0
        adv_x = temp + totalchangge
        adv_x_0 = tf.concat([adv_x_0, adv_x], 0)
        totalchangge_0 = tf.concat([totalchangge_0, totalchangge], 0)
        i = tf.add(i, 1)
    adv_x_0 = tf.clip_by_value(adv_x_0, 0.0, 1.0)
    # adv_x_0_var = tf.Variable(adv_x_0)

    for ii in range(1, steps):
        i = tf.constant(0)

        while (tf.less(i, numberofcommonindex)):
            temp = adv_x_0[i]
            temp = tf.expand_dims(temp, 0)
            change_temp = totalchangge_0[i]
            change_temp = tf.expand_dims(change_temp, 0)
            temp_y = y_test_binary_ndarray[common_index[i, 0]]
            temp_y = tf.expand_dims(temp_y, 0)

            perturbations = create_adversarial_pattern(temp, temp_y)
            if tf.reduce_max(perturbations) == tf.reduce_min(perturbations):
                assert tf.reduce_max(perturbations) == 0
            totalchangge = stepsize * perturbations/255.0
            change_temp = change_temp + totalchangge
            change_temp = tf.clip_by_value(change_temp, -epsilons/255.0, epsilons/255.0)
            temp = x_test[common_index[i, 0]]
            temp = tf.expand_dims(temp, 0)
            adv_x = temp + change_temp

            adv_x_0 = tf.tensor_scatter_nd_update(adv_x_0, [[i]], adv_x)
            totalchangge_0 = tf.tensor_scatter_nd_update(totalchangge_0, [[i]], change_temp)
            i = tf.add(i, 1)
        adv_x_0 = tf.clip_by_value(adv_x_0, 0.0, 1.0)

    score = model_binary.evaluate(adv_x_0, y_test_binary_ndarray_common, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    score[1] = (numberofcommonindex * score[1] - zerogra) / (numberofcommonindex - zerogra)
    # save loss and accuracy, append mode 2023-6-4
    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/PCL/'
    namestring = 'weiVGGpureModel'
    namestring += datetime
    namestring += '-cifar10-repeat'

    headerstring = namestring
    namestring += '-pgd'
    namestring += str(epsilons)
    namestring += '-'
    namestring += str(stepsize)
    namestring += '-'
    namestring += str(steps)
    namestring += '.txt'

    with open(pathstring + namestring, 'a') as f:
        np.savetxt(f, score, delimiter=",", header=headerstring, fmt='%.4f', newline=' ')
        f.write("\n")
    """
    #cw
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    model_binary.call(inputs)

    
    # model_concat = keras.Model(inputs=model_binary.layers[0].input, outputs=model_binary.get_layer('flatten').output)
    model_concat = keras.Model(inputs=model_binary.layers[0].input, outputs=model_binary.get_layer('dense_2').output)

    CW = CarliniL2(model=model_concat)

    temp = x_test[common_index[0, 0].numpy()]
    temp = tf.expand_dims(temp, 0)
    temp_y = y_test_binary_ndarray[common_index[0, 0].numpy()]
    temp_y = tf.expand_dims(temp_y, 0)
    y_test_binary_ndarray_common = temp_y

    x_test_common = temp
    i = tf.constant(1)
    while (tf.less(i, numberofcommoncw)):
        temp = x_test[common_index[i, 0]]
        temp = tf.expand_dims(temp, 0)
        temp_y = y_test_binary_ndarray[common_index[i, 0]]
        temp_y = tf.expand_dims(temp_y, 0)
        y_test_binary_ndarray_common = tf.concat([y_test_binary_ndarray_common, temp_y], 0)
        x_test_common = tf.concat([x_test_common, temp], 0)
        i = tf.add(i, 1)

    dis = CW.attack(x_test_common, y_test_binary_ndarray_common)

    min_disl2 = tf.reduce_min(dis)
    max_disl2 = tf.reduce_max(dis)
    ave_disl2 = tf.reduce_mean(dis)
    dis_value = np.array([ave_disl2, max_disl2, min_disl2])

    # save disl2, append mode 2023-6-4
    pathstring = '/mnt/ex/PycharmProjects/BinaryFeatures/dataset-cifar10/'
    namestring = 'weiVGGpureModel'
    namestring += datetime
    namestring += '-repeat'

    headerstring = namestring
    namestring += '-cw'
    namestring += str(numberofcommoncw)

    namestring += '.txt'

    with open(pathstring + namestring, 'a') as f:
        np.savetxt(f, dis_value, delimiter=",", header=headerstring, fmt='%.4f', newline=' ')
        f.write("\n")
    """
    return

import sys
if __name__ == '__main__':
    argv1 = int(sys.argv[1])
    argv2 = int(sys.argv[2])
    argv4 = int(sys.argv[4])
    argv5 = int(sys.argv[5])
    argv6 = int(sys.argv[6])
    argv7 = int(sys.argv[7])
    comp_cifar10_vgg_pure_PCL(numberofcommon=argv1, repeat=argv2, datetime=sys.argv[3], epsilons=argv4, stepsize=argv5, steps=argv6, numberofcommoncw=argv7)



