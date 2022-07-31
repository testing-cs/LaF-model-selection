import argparse
import os
from utils import dataSel
from config import hyperparameters
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataName', default='java250', type=str, choices=['mnist', 'cifar10', 'fashion', 'amazon', 'iwildcamO', 'java250', 'C++1000'])
    parser.add_argument('--metric', default="random", type=str, choices=["sds", "random", "ces"])
    return parser.parse_args()


def main():
    args = get_args()
    dataName = args.dataName
    metric = args.metric
    if dataName in ['mnist', 'cifar10']:
        dataTypes = ['original', 'Gaussian_Noise', 'Shot_Noise', 'Impulse_Noise', 'Defocus_Blur', 'Glass_Blur',
                     'Zoom_Blur', 'Snow', 'Fog', 'Brightness', 'Contrast', 'Elastic', 'JPEG', 'Pixelate', 'Frost', 'Motion_Blur']
    elif dataName in ["fashion", "C++1000"]:
        dataTypes = ['original']
    elif dataName == 'java250':
        dataTypes = ['original', 'ood']
    else:
        dataTypes = ["id", "ood"]
    for dataType in dataTypes:
        if dataType in ["original", "id", "ood"]:
            severities = [0]
        else:
            severities = range(1, 6)
        for severity in severities:
            parameters = hyperparameters(dataName, dataType, severity)
            if metric in ["sds", "random", "ces"]:
                iteNum = 50
            else:
                iteNum = 1
            for ite in range(iteNum):
                for num in range(parameters.model_num, 185, 5):
                    dataSel(parameters, metric, num, ite)


if __name__ == '__main__':
    main()
