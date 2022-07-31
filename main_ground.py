import gc
import numpy as np
import keras.backend as K
import argparse
from utils_ground import load_data, load_models, pred_read
from config import hyperparameters


def computeAccuracy(model, parameters, data, labels):
    split_num = int(np.ceil(len(data) / parameters.batch_size))
    accuracy = 0
    for i in range(split_num):
        batch_data = data[i * parameters.batch_size:(i + 1) * parameters.batch_size]
        prediction_batch = model.predict(batch_data)
        accuracy += np.sum(np.argmax(prediction_batch, axis=1) == labels[i * parameters.batch_size:(i + 1) * parameters.batch_size])

    return accuracy / len(labels)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataName', default='mnist', type=str, choices=['mnist', 'fashion', 'cifar10'])
    return parser.parse_args()


def main():
    global _, x_test, y_test
    args = get_args()
    dataName = args.dataName
    if dataName in ['mnist', 'cifar10']:
        dataTypes = ['original', 'Gaussian_Noise', 'Shot_Noise', 'Impulse_Noise', 'Defocus_Blur', 'Glass_Blur',
                     'Zoom_Blur', 'Snow', 'Fog', 'Brightness', 'Contrast', 'Elastic', 'JPEG', 'Pixelate', 'Frost', 'Motion_Blur']
    else:
        dataTypes = ['original']
    for dataType in dataTypes:
        if dataType in ["original"]:
            severities = [0]
        else:
            severities = range(1, 6)
        for severity in severities:
            print("{0}-{1}".format(dataType, severity))
            parameters = hyperparameters(dataName, dataType, severity)
            x_test, y_test = load_data(parameters)
            # save_name = parameters.save_ground_root + "ground-{0}-{1}.npy".format(dataType, severity)
            model = load_models(parameters)
            pred_read(model, x_test, parameters)
            # label_list = label_read(model, x_test, y_test, parameters)
            # print(label_list.shape)
            K.clear_session()
            del model
            gc.collect()
            # accuracy = computeAcc(parameters, label_list, np.arange(label_list.shape[0]), label_list.shape[0])
            # np.save(save_name, accuracy)
            # print(accuracy)


if __name__ == '__main__':
    main()
