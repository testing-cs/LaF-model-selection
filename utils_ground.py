import numpy as np
import glob
from keras.models import load_model
from keras.datasets import fashion_mnist, mnist, cifar10


def load_data_split(parameters):
    try:
        if parameters.dataName == "fashion":
            (_, _), (x_test, y_test) = fashion_mnist.load_data()
            if parameters.dataType != "original-split":
                x_test = np.load(parameters.save_data_root_adv + "{0}-{1}.npy".format(parameters.dataType.replace('-split', ''), parameters.severity))
            img_rows, img_cols = 28, 28
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            x_test = x_test.astype('float32') / 255
        elif parameters.dataName == "mnist":
            (_, _), (x_test, y_test) = mnist.load_data()
            if parameters.dataType != "original-split":
                x_test = np.load(parameters.save_data_root_adv + "{0}-{1}.npy".format(parameters.dataType.replace('-split', ''), parameters.severity))
            img_rows, img_cols = 28, 28
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            x_test = x_test.astype('float32') / 255
        elif parameters.dataName == "cifar10":
            (_, _), (x_test, y_test) = cifar10.load_data()
            if parameters.dataType != "original-split":
                x_test = np.load(parameters.save_data_root_adv + "{0}-{1}.npy".format(parameters.dataType.replace('-split', ''), parameters.severity))
            x_test = x_test.astype('float32') / 255
            y_test = np.squeeze(y_test)
        x_test_first = x_test[:5000, :]
        y_test_first = y_test[:5000]
        x_test_second = x_test[5000:, :]
        y_test_second = y_test[5000:]
        return x_test_first, y_test_first, x_test_second, y_test_second
    except:
        print("invalid data name")


def load_data(parameters):
    try:
        if parameters.dataName == "fashion":
            (_, _), (x_test, y_test) = fashion_mnist.load_data()
            if parameters.dataType != "original":
                x_test = np.load(parameters.save_data_root_adv + "{0}-{1}.npy".format(parameters.dataType, parameters.severity))
            img_rows, img_cols = 28, 28
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            x_test = x_test.astype('float32') / 255
        elif parameters.dataName == "mnist":
            (_, _), (x_test, y_test) = mnist.load_data()
            if parameters.dataType != "original":
                x_test = np.load(parameters.save_data_root_adv + "{0}-{1}.npy".format(parameters.dataType, parameters.severity))
            img_rows, img_cols = 28, 28
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            x_test = x_test.astype('float32') / 255
        elif parameters.dataName == "cifar10":
            (_, _), (x_test, y_test) = cifar10.load_data()
            if parameters.dataType != "original":
                x_test = np.load(parameters.save_data_root_adv + "{0}-{1}.npy".format(parameters.dataType, parameters.severity))
            x_test = x_test.astype('float32') / 255
            y_test = np.squeeze(y_test)
        return x_test, y_test
    except:
        print("invalid data name")


def load_models(parameters):
    model = []
    model_name = '{0}/model-'.format(parameters.save_model_root)
    modelNum = len(glob.glob1(parameters.save_model_root, "*.h5"))
    filename = []
    for i in range(modelNum):
        model_name_ = model_name + str(i) + '.h5'
        filename.append(model_name_)

    for i in range(len(filename)):
        model.append(load_model(filename[i]))
    return model


def computeAcc(parameters, label_list, select_index, num):
    acc_list = np.zeros(parameters.model_num)
    for j in range(parameters.model_num):
        acc_list[j] = np.sum(label_list[select_index, j+1] == label_list[select_index, 0]) / num
    return acc_list


def label_read(model, x_test, y_test, parameters):
    label_list = np.zeros((len(x_test), len(model) + 1))
    label_list[:, 0] = y_test
    for i in range(len(model)):
        label_list[:, i+1] = np.argmax(model[i].predict(x_test), axis=1)
    np.save(parameters.save_ground_root + "labels-{0}-{1}.npy".format(parameters.dataType, parameters.severity), label_list)
    return label_list


def pred_read(model, x_test, parameters):
    for i in range(len(model)):
        model_pre = model[i].predict(x_test)
        np.save(parameters.save_model_pre_root + "pre-{0}-{1}-{2}.npy".format(i, parameters.dataType, parameters.severity), model_pre)


