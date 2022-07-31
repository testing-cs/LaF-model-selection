import argparse
import numpy as np
from config import hyperparameters
from utils_laf import em


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataName', default='java250', type=str, choices=['mnist', 'cifar10', 'fashion', 'amazon', 'iwildcamO', 'java250', 'C++1000'])
    parser.add_argument('--dataType', default='ood', type=str)
    parser.add_argument('--method', default='new', type=str, choices=['new'])  # "new" is laf
    parser.add_argument('--ite', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    dataName = args.dataName
    print(dataName)
    dataType = args.dataType
    if dataType in ["original", "id", "ood"]:
        severities = [0]
    else:
        severities = range(1, 6)
    for severity in severities:
        parameters = hyperparameters(dataName, dataType, severity)
        if args.ite == 0:
            saveName = parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npz".format(args.method, 0, parameters.dataType, parameters.severity, 0)
        else:
            saveName = parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}-{5}.npz".format(args.method, 0, parameters.dataType, parameters.severity, 0, args.ite)
        label_list = np.load(parameters.save_ground_root + "labels-{0}-{1}.npy".format(parameters.dataType, parameters.severity))
        candidate_index = []
        for rowNo in range(len(label_list)):
            if len(np.unique(label_list[rowNo, 1:])) > 1:
                candidate_index.append(rowNo)
        label_list_new = label_list[candidate_index, :].astype(int)
        input_file = parameters.save_result_root + '{0}-{1}-glad-filter.txt'.format(dataType, severity)
        with open(input_file, 'w') as the_file:
            for lineNo in range(len(label_list_new) + 1):
                if lineNo == 0:
                    the_file.write("{0} {1} {2} {3}\n".format(parameters.model_num * len(label_list_new), parameters.model_num, len(label_list_new),
                                                              parameters.class_num))
                else:
                    for modelId in range(parameters.model_num):
                        the_file.write("{0} {1} {2}\n".format(lineNo - 1, modelId, label_list_new[lineNo - 1, modelId + 1]))
        glad_accuracy, glad_easyness, glad_label = em(input_file, label_list=label_list_new[:, 1:])
        metric_acc = glad_accuracy[:, 1]
        np.savez(saveName, x=metric_acc, y=None)


if __name__ == '__main__':
    main()
